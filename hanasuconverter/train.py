import os
import argparse
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import csv # Import csv

# Assuming these imports exist and are correct
from hanasuconverter import commons # Import commons explicitly
from hanasuconverter.commons import kl_divergence
from hanasuconverter.models import SynthesizerTrn
from hanasuconverter.audio import mel_spectrogram, load_wav_to_torch
from hanasuconverter.utils import (
    get_hparams_from_file,
    save_checkpoint,
    load_checkpoint, # Import load_checkpoint
    scan_checkpoint,
    list_audio_files, # Import list_audio_files
    HParams # Import HParams if used for type hints etc.
)
import torch.amp # Import amp for autocast

class VoiceDataset(torch.utils.data.Dataset):
    """
    Dataset for voice conversion training
    Loads audio, converts to mono, calculates mel, returns random segment.
    """
    def __init__(self, data_list, hparams):
        self.data_list = data_list # List of (audio_path, speaker_name)
        self.hps = hparams
        self.segment_size = hparams.train.segment_size # Waveform segment size
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.n_mel_channels = hparams.data.n_mel_channels
        self.mel_fmin = hparams.data.mel_fmin
        self.mel_fmax = hparams.data.mel_fmax

        # Create speaker mapping (name to integer ID)
        self.speakers = sorted(list(set([x[1] for x in data_list])))
        self.speaker_to_id = {speaker: i for i, speaker in enumerate(self.speakers)}
        print(f"Dataset contains {len(self.speakers)} speakers: {self.speakers}")
        print(f"Speaker mapping: {self.speaker_to_id}")

    def __getitem__(self, index):
        audio_path, speaker_name = self.data_list[index]
        speaker_id = self.speaker_to_id[speaker_name]

        # Load audio (target sample rate)
        # load_wav_to_torch returns FloatTensor
        audio, sr = load_wav_to_torch(audio_path, self.sampling_rate)

        # Handle stereo audio (convert to mono for mel spec and length calc)
        # Keep original dimensionality for returning 'x' if needed by model (though it's usually unused)
        audio_for_mel = audio
        if audio.dim() > 1 and audio.shape[0] > 1: # Check if stereo
            audio_for_mel = torch.mean(audio, dim=0) # Convert to mono for mel

        # Ensure audio is long enough for segment
        if audio_for_mel.size(0) < self.segment_size:
            # Pad if too short (use reflect padding or constant 0)
            pad_size = self.segment_size - audio_for_mel.size(0)
            # Pad mono version for mel calculation
            audio_for_mel = F.pad(audio_for_mel, (0, pad_size), 'constant', 0)
            # Pad original audio (could be stereo)
            audio = F.pad(audio, (0, pad_size), 'constant', 0) # Pad last dimension (time)
        elif audio_for_mel.size(0) > self.segment_size:
             # Random segment for training
             max_start = audio_for_mel.size(0) - self.segment_size
             start = torch.randint(0, max_start + 1, (1,)).item()
             # Slice mono version for mel
             audio_for_mel = audio_for_mel[start:start + self.segment_size]
             # Slice original audio (could be stereo)
             audio = audio[..., start:start + self.segment_size] # Slice last dimension (time)
        # If exactly segment_size, no slicing/padding needed

        # Generate mel spectrogram from the (potentially padded/sliced) mono audio
        # Input to mel_spectrogram needs shape [B, T], so add batch dim
        mel = mel_spectrogram(
            audio_for_mel.unsqueeze(0),
            self.filter_length,
            self.n_mel_channels,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            self.mel_fmin,
            self.mel_fmax
        ).squeeze(0) # Remove batch dim, result shape [n_mels, T_mel]

        # Return waveform segment (x), mel segment (y), speaker_id
        # Ensure waveform 'x' has channel dimension if model expects it (e.g., [C, T])
        # Our model currently uses mel 'y', so 'x' might just be a placeholder.
        # Return stereo if possible, otherwise mono with channel dim.
        if audio.dim() == 1: # If was mono originally or became mono
             audio = audio.unsqueeze(0) # Add channel dim: [1, T]

        # x: [C, segment_size], y: [n_mels, mel_segment_size], speaker_id: int
        return audio, mel, speaker_id

    def __len__(self):
        return len(self.data_list)

def train_step(model, optimizer, scaler, x, x_lengths, y, y_lengths, sid_src, sid_tgt, hps):
    """
    Perform a single training step
    """
    # Get device from model parameters
    device = next(model.parameters()).device

    # Move data to device
    x, x_lengths = x.to(device, non_blocking=True), x_lengths.to(device, non_blocking=True)
    y, y_lengths = y.to(device, non_blocking=True), y_lengths.to(device, non_blocking=True)
    sid_src = sid_src.to(device, non_blocking=True)
    sid_tgt = sid_tgt.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True) # More efficient zeroing

    # --- Calculate random segment indices for loss ---
    mel_segment_size = hps.train.segment_size // hps.data.hop_length
    wav_segment_size = hps.train.segment_size

    # Ensure lengths are sufficient for slicing (y_lengths is length of mel spec)
    min_len = torch.min(y_lengths)
    if min_len < mel_segment_size:
         print(f"Warning: Min y_length ({min_len}) < mel_segment_size ({mel_segment_size}). Required wav length {wav_segment_size}. Skipping batch.")
         return None # Skip this batch if any sample is too short

    ids_slice = (torch.rand(y.size(0)).to(y.device) * (y_lengths.float() - mel_segment_size + 1)).long()
    wav_ids_slice = ids_slice * hps.data.hop_length
    # --------------------------------------------

    # Use autocast for mixed precision (handles CPU/MPS gracefully)
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if hps.train.fp16_run and device.type == 'cuda' else torch.float32, enabled=hps.train.fp16_run and device.type != 'cpu'):
        # Model forward pass
        # y_hat: [B, 2, T_wav], z_mask: [B, 1, T_mel]
        y_hat, _, z_mask, \
        (z, z_p, m_p, logs_p, m_q, logs_q) = model(x, x_lengths, y, y_lengths, sid_src=sid_src, sid_tgt=sid_tgt)

        # --- Compute losses ---
        # Slice target mel spectrogram using generated mel ids_slice
        # Input y: [B, n_mels, T_mel], output y_mel: [B, n_mels, mel_segment_size]
        y_mel = commons.slice_segments(y, ids_slice, mel_segment_size)

        # Slice predicted waveform y_hat based on the *corresponding* waveform segment indices
        # Input y_hat: [B, 2, T_wav], output y_hat_slice: [B, 2, wav_segment_size]
        y_hat_slice = commons.slice_segments(y_hat, wav_ids_slice, wav_segment_size)

        # Convert sliced predicted waveform to mel spectrogram
        # Input y_hat_slice: [B, 2, wav_segment_size] -> mean to mono -> [B, wav_segment_size]
        # Output y_hat_mel: [B, n_mels, mel_segment_size]
        y_hat_mono_slice = torch.mean(y_hat_slice, dim=1) # Convert stereo output to mono for mel loss
        y_hat_mel = mel_spectrogram(y_hat_mono_slice, # Use sliced mono waveform
                                   hps.data.filter_length,
                                   hps.data.n_mel_channels,
                                   hps.data.sampling_rate,
                                   hps.data.hop_length,
                                   hps.data.win_length,
                                   hps.data.mel_fmin,
                                   hps.data.mel_fmax)

        # Reconstruction loss (L1 on mel spectrograms)
        mel_loss = F.l1_loss(y_hat_mel, y_mel) * hps.train.c_mel

        # KL divergence loss (Calculated on full sequences, masked by z_mask)
        # Ensure z_mask has the correct shape [B, 1, T_mel] matching m_q etc.
        kl_loss = kl_divergence(m_q, logs_q, m_p, logs_p) # Removed unnecessary sum/mean here
        kl_loss = torch.sum(kl_loss / torch.sum(z_mask)) * hps.train.c_kl # Normalize by mask sum

        # Total loss
        total_loss = mel_loss + kl_loss

    # Backward pass using scaler (scaler handles enabled=False internally if not CUDA/FP16)
    scaler.scale(total_loss).backward()

    # Gradient clipping (unscale first)
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.train.grad_clip)

    # Update weights using scaler
    scaler.step(optimizer)
    scaler.update()

    # Return metrics
    grad_norm_item = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm # Handle tensor/float grad_norm
    return {
        'total_loss': total_loss.item(),
        'mel_loss': mel_loss.item(),
        'kl_loss': kl_loss.item(),
        'grad_norm': grad_norm_item
    }

def train(rank, hps, port):
    """
    Main training function
    """
    # Initialize distributed training if needed
    if hps.train.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend='nccl', init_method=f'env://', world_size=hps.train.num_gpus, rank=rank)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Rank {rank} using device: {device}")
    torch.manual_seed(hps.train.seed + rank) # Ensure different seeds per process if distributed

    # Create dataset BEFORE model to get n_speakers
    # hps.data.training_files should be populated in main()
    if not hasattr(hps.data, 'training_files') or not hps.data.training_files:
         if rank == 0: print("Error: Training file list is empty.")
         return
    train_dataset = VoiceDataset(hps.data.training_files, hps)
    n_speakers = len(train_dataset.speakers) # Get number of speakers
    if rank == 0: print(f"Determined {n_speakers} speakers from dataset.")

    # Create model (passing n_speakers)
    model = SynthesizerTrn(
        hps.data.n_mel_channels,         # Use n_mel_channels
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        n_speakers=n_speakers,           # <-- Pass n_speakers
        gin_channels=hps.model.gin_channels,
        use_external_speaker_embedding=False # Training uses internal embeddings
    ).to(device) # Move model to device

    # Wrap model for distributed training if needed
    if hps.train.distributed:
        # find_unused_parameters might be needed if some params aren't used in forward
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
        weight_decay=hps.train.weight_decay
    )

    # Initialize GradScaler (works correctly even if FP16 is disabled or device is not CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=hps.train.fp16_run and device.type == 'cuda')

    # Load checkpoint if continuing training
    global_step = 0
    start_epoch = 0
    if hps.train.continue_from_checkpoint:
        checkpoint_path = hps.train.checkpoint_path
        # Use scan_checkpoint to find the latest if path is a directory
        if os.path.isdir(checkpoint_path):
            actual_checkpoint_path = scan_checkpoint(checkpoint_path, 'model_') # Find latest like model_********.pt
            if not actual_checkpoint_path:
                 print(f"Warning: Checkpoint directory specified ({checkpoint_path}), but no checkpoints found. Starting fresh.")
                 actual_checkpoint_path = None
            else:
                 print(f"Found latest checkpoint in directory: {actual_checkpoint_path}")
        elif os.path.isfile(checkpoint_path):
             actual_checkpoint_path = checkpoint_path # Use specified file
        else:
             print(f"Warning: Checkpoint path specified ({checkpoint_path}) but not found. Starting fresh.")
             actual_checkpoint_path = None

        if actual_checkpoint_path:
            # Load checkpoint using the utility function
            model_to_load = model.module if hps.train.distributed else model
            try:
                model_to_load, optimizer, _, iteration, scaler = load_checkpoint(
                    actual_checkpoint_path, model_to_load, optimizer, scaler
                )
                global_step = iteration
                # Calculate start epoch based on loaded step and dataloader length
                # Need dataloader length first, calculate later
                print(f"Loaded checkpoint at iteration {global_step}")
            except Exception as e:
                print(f"Error loading checkpoint {actual_checkpoint_path}: {e}. Starting fresh.")
                global_step = 0 # Reset step if loading failed
    else:
        print("Starting training from scratch (no checkpoint specified).")

    # Create dataloader (conditional pin_memory, get len for epoch calc)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=hps.train.num_gpus,
        rank=rank,
        shuffle=True
    ) if hps.train.distributed else None

    # Use drop_last=True to ensure consistent batch sizes, important for epoch calculation
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=hps.train.num_workers,
        sampler=train_sampler,
        pin_memory=device.type == 'cuda', # Conditional pin_memory
        drop_last=True # Drop last incomplete batch
    )

    # Calculate start epoch now that we have dataloader length
    if global_step > 0:
         # Ensure len(train_loader) > 0 to avoid division by zero
         if len(train_loader) > 0:
              start_epoch = global_step // len(train_loader)
         else:
              if rank == 0: print("Warning: Train loader length is 0, cannot calculate start epoch.")
              start_epoch = 0 # Default to 0 if loader is empty
    else:
         start_epoch = 0

    if rank == 0: print(f"Starting training from Epoch {start_epoch + 1}")

    # Create scheduler AFTER loading checkpoint and calculating start_epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=hps.train.lr_decay,
        last_epoch=global_step - 1 # Resume scheduler correctly from the step before the current one
    )

    # Create validation dataset if specified (only rank 0 needs it)
    val_loader = None
    if hps.data.validation_files and rank == 0:
        val_data_list = prepare_datalist(hps.data.validation_files) # Assuming prepare_datalist works
        if val_data_list:
             val_dataset = VoiceDataset(val_data_list, hps)
             val_loader = DataLoader(
                 val_dataset,
                 batch_size=hps.train.batch_size, # Can use smaller batch size for validation if needed
                 shuffle=False,
                 num_workers=hps.train.num_workers,
                 pin_memory=device.type == 'cuda', # Conditional pin_memory
                 drop_last=False # Don't drop last validation batch
             )
             print(f"Validation dataset loaded with {len(val_dataset)} samples.")
        else:
             print("Warning: Validation files specified but no files found or processed.")

    # Create tensorboard writer and checkpoint directory only on rank 0
    writer = None
    if rank == 0:
        os.makedirs(hps.train.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=hps.train.log_dir)
        os.makedirs(hps.train.checkpoint_dir, exist_ok=True)

    # Training loop
    model.train()
    for epoch in range(start_epoch, hps.train.epochs):
        if rank == 0:
            print(f"--- Starting Epoch {epoch+1}/{hps.train.epochs} ---")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training iterations
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hps.train.epochs}", disable=(rank != 0))
        for batch_idx, batch in enumerate(pbar):
            # Get batch data (x:[B,C,T_wav], y:[B,M,T_mel], speaker_id:[B])
            x, y, speaker_id = batch
            # Get lengths from segment sizes defined in hps
            x_lengths = torch.full((x.size(0),), hps.train.segment_size, dtype=torch.long) # Waveform length
            y_lengths = torch.full((y.size(0),), hps.train.segment_size // hps.data.hop_length, dtype=torch.long) # Mel length

            # Create source and target speaker IDs based on training strategy
            batch_size = x.size(0)
            if n_speakers > 1 and np.random.random() < hps.train.same_speaker_prob:
                sid_src = speaker_id
                sid_tgt = speaker_id
            elif n_speakers > 1:
                sid_src = speaker_id
                sid_tgt_list = []
                possible_targets = list(range(n_speakers))
                for sid in sid_src.tolist(): # Iterate over python list
                    targets = [i for i in possible_targets if i != sid]
                    # targets should not be empty if n_speakers > 1
                    sid_tgt_list.append(np.random.choice(targets))
                sid_tgt = torch.tensor(sid_tgt_list, dtype=torch.long)
            else: # Only one speaker, reconstruction only
                 sid_src = speaker_id
                 sid_tgt = speaker_id

            # Train step
            metrics = train_step(model, optimizer, scaler, x, x_lengths, y, y_lengths, sid_src, sid_tgt, hps)

            # Log metrics and update progress bar only on rank 0
            if rank == 0:
                if metrics: # Check if batch was skipped
                    # Format metrics for display
                    postfix_dict = {k: f"{v:.4f}" for k, v in metrics.items()}
                    postfix_dict["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
                    postfix_dict["step"] = global_step
                    pbar.set_postfix(postfix_dict)

                    if global_step % hps.train.log_interval == 0:
                        lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar("train/learning_rate", lr, global_step)
                        for k, v in metrics.items():
                            writer.add_scalar(f"train/{k}", v, global_step)
                        writer.add_scalar("memory/allocated_gb", torch.cuda.memory_allocated(device) / (1024**3) if device.type=='cuda' else 0, global_step)
                        writer.add_scalar("memory/reserved_gb", torch.cuda.memory_reserved(device) / (1024**3) if device.type=='cuda' else 0, global_step)

                # Save checkpoint periodically
                if global_step > 0 and global_step % hps.train.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(hps.train.checkpoint_dir, f"model_{global_step:08d}.pt")
                    save_checkpoint(
                        model.module if hps.train.distributed else model,
                        optimizer,
                        optimizer.param_groups[0]['lr'],
                        global_step,
                        checkpoint_path,
                        scaler # Save scaler state
                    )
                    # Optionally save latest checkpoint separately for easy resuming
                    latest_path = os.path.join(hps.train.checkpoint_dir, "model_latest.pt")
                    save_checkpoint(
                        model.module if hps.train.distributed else model,
                        optimizer,
                        optimizer.param_groups[0]['lr'],
                        global_step,
                        latest_path,
                        scaler
                    )

            global_step += 1

        # --- End of Epoch ---

        # Update learning rate scheduler *after* each epoch
        # Ensure scheduler step happens after optimizer step within the epoch logic if needed,
        # but typically ExponentialLR steps per epoch.
        scheduler.step()

        # Validation (run only on rank 0 after each epoch)
        if val_loader is not None and rank == 0:
            model.eval()
            val_losses = {'total': [], 'mel': [], 'kl': []}
            print("\n--- Starting Validation ---")
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                    x, y, speaker_id = batch
                    # Use full segment lengths for validation consistency
                    x_lengths = torch.full((x.size(0),), hps.train.segment_size, dtype=torch.long)
                    y_lengths = torch.full((y.size(0),), hps.train.segment_size // hps.data.hop_length, dtype=torch.long)

                    # Use same speaker for validation (reconstruction)
                    sid_src = speaker_id
                    sid_tgt = speaker_id

                    # Move data to device
                    x, x_lengths = x.to(device), x_lengths.to(device)
                    y, y_lengths = y.to(device), y_lengths.to(device)
                    sid_src = sid_src.to(device)
                    sid_tgt = sid_tgt.to(device)

                    # Autocast for validation inference
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if hps.train.fp16_run and device.type == 'cuda' else torch.float32, enabled=hps.train.fp16_run and device.type != 'cpu'):
                        # Model forward pass
                        y_hat, _, z_mask, \
                        (z, z_p, m_p, logs_p, m_q, logs_q) = model(x, x_lengths, y, y_lengths, sid_src=sid_src, sid_tgt=sid_tgt)

                        # Compute losses on a random slice for validation consistency
                        mel_segment_size = hps.train.segment_size // hps.data.hop_length
                        wav_segment_size = hps.train.segment_size

                        min_len_val = torch.min(y_lengths)
                        if min_len_val < mel_segment_size:
                             print(f"Warning: Skipping validation batch, min length {min_len_val} < {mel_segment_size}")
                             continue # Skip if too short

                        # Use fixed start for validation slice or random? Random is fine.
                        ids_slice = (torch.rand(y.size(0)).to(y.device) * (y_lengths.float() - mel_segment_size + 1)).long()
                        wav_ids_slice = ids_slice * hps.data.hop_length

                        y_mel = commons.slice_segments(y, ids_slice, mel_segment_size)
                        y_hat_slice = commons.slice_segments(y_hat, wav_ids_slice, wav_segment_size)
                        y_hat_mono_slice = torch.mean(y_hat_slice, dim=1) # Mono for mel calc

                        y_hat_mel = mel_spectrogram(y_hat_mono_slice,
                                                   hps.data.filter_length,
                                                   hps.data.n_mel_channels,
                                                   hps.data.sampling_rate,
                                                   hps.data.hop_length,
                                                   hps.data.win_length,
                                                   hps.data.mel_fmin,
                                                   hps.data.mel_fmax)

                        mel_loss = F.l1_loss(y_hat_mel, y_mel) * hps.train.c_mel
                        kl_loss = kl_divergence(m_q, logs_q, m_p, logs_p, z_mask)
                        kl_loss = torch.sum(kl_loss / torch.sum(z_mask)) * hps.train.c_kl

                        total_loss = mel_loss + kl_loss
                        val_losses['total'].append(total_loss.item())
                        val_losses['mel'].append(mel_loss.item())
                        val_losses['kl'].append(kl_loss.item())

            # Log validation metrics if validation was performed
            if val_losses['total']:
                avg_val_loss = sum(val_losses['total']) / len(val_losses['total'])
                avg_mel_loss = sum(val_losses['mel']) / len(val_losses['mel'])
                avg_kl_loss = sum(val_losses['kl']) / len(val_losses['kl'])
                writer.add_scalar("val/total_loss", avg_val_loss, global_step)
                writer.add_scalar("val/mel_loss", avg_mel_loss, global_step)
                writer.add_scalar("val/kl_loss", avg_kl_loss, global_step)
                print(f"Validation Epoch {epoch+1} Loss: {avg_val_loss:.4f} (Mel: {avg_mel_loss:.4f}, KL: {avg_kl_loss:.4f})\n")
            else:
                 print(f"Validation Epoch {epoch+1}: No validation batches processed.\n")

            model.train() # Set back to train mode

    # --- End of Training Loop ---

    # Save final checkpoint only on rank 0
    if rank == 0:
        checkpoint_path = os.path.join(hps.train.checkpoint_dir, f"model_final.pt")
        save_checkpoint(
            model.module if hps.train.distributed else model,
            optimizer,
            optimizer.param_groups[0]['lr'],
            global_step,
            checkpoint_path,
            scaler
        )
        print(f"Saved final checkpoint to {checkpoint_path}")
        if writer:
             writer.close()

    # Clean up distributed processes
    if hps.train.distributed:
        dist.destroy_process_group()

def prepare_datalist(data_path, output_path=None):
    """
    Prepare data list for training: (audio_path, speaker_name) tuples

    Args:
        data_path: Path to data directory (e.g., data/speaker1/file.wav) or CSV file
        output_path: Path to save data list CSV (optional)

    Returns:
        List of (audio_path, speaker_name) tuples
    """
    data_list = []
    supported_extensions = ['.wav', '.mp3', '.flac', '.ogg']

    # If data_path is a CSV file
    if os.path.isfile(data_path) and data_path.lower().endswith('.csv'):
        print(f"Reading data list from CSV: {data_path}")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Read header

                # Find column indices (case-insensitive)
                header_lower = [h.lower().strip() for h in header]
                try:
                    path_idx = header_lower.index('path')
                    speaker_idx = header_lower.index('speaker_id') # Or 'speaker'
                except ValueError:
                    print("Error: CSV must contain 'path' and 'speaker_id' columns.")
                    return []

                for i, row in enumerate(reader):
                    if len(row) > max(path_idx, speaker_idx):
                        audio_path = row[path_idx].strip()
                        speaker_id_or_name = row[speaker_idx].strip()

                        # Basic validation
                        if os.path.exists(audio_path) and any(audio_path.lower().endswith(ext) for ext in supported_extensions):
                           data_list.append((audio_path, speaker_id_or_name))
                        else:
                            print(f"Warning: Skipping invalid entry in CSV row {i+2}: Path='{audio_path}', Speaker='{speaker_id_or_name}'")
                    else:
                        print(f"Warning: Skipping malformed CSV row {i+2}")
        except Exception as e:
            print(f"Error reading CSV file {data_path}: {e}")
            return []

    # If data_path is a directory (expecting structure like data_path/speaker_name/audio.wav)
    elif os.path.isdir(data_path):
        print(f"Scanning data directory: {data_path}")
        audio_files = list_audio_files(data_path, extensions=supported_extensions)
        for audio_path in audio_files:
            try:
                # Extract speaker name from the parent directory name
                speaker_name = os.path.basename(os.path.dirname(audio_path))
                if speaker_name: # Ensure speaker name is not empty
                    data_list.append((audio_path, speaker_name))
                else:
                    print(f"Warning: Could not determine speaker name for {audio_path}")
            except Exception as e:
                print(f"Warning: Error processing path {audio_path}: {e}")

    else:
        print(f"Error: data_path '{data_path}' is not a valid CSV file or directory.")
        return []

    # Save data list to CSV if output_path is provided
    if output_path is not None:
        print(f"Saving data list to CSV: {output_path}")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['path', 'speaker_id']) # Use 'speaker_id' as standard header
                writer.writerows(data_list)
        except Exception as e:
            print(f"Error writing data list CSV to {output_path}: {e}")

    print(f"Prepared data list with {len(data_list)} audio files.")
    return data_list

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Hanasu Converter Training Script")
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data directory or CSV file')
    parser.add_argument('--output_dir', type=str, default='./hanasu_output', help='Base directory to save logs and checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, help='Specific directory for checkpoints (overrides output_dir/checkpoints)')
    parser.add_argument('--log_dir', type=str, help='Specific directory for logs (overrides output_dir/logs)')
    parser.add_argument('--checkpoint_path', type=str, help='Path to specific checkpoint file or directory containing checkpoints to continue training')
    parser.add_argument('--validation_files', type=str, help='Path to validation data CSV or directory (optional)')
    parser.add_argument('--epochs', type=int, help='Override number of training epochs from config')
    parser.add_argument('--batch_size', type=int, help='Override batch size from config')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate from config')
    parser.add_argument('--seed', type=int, help='Override random seed from config')
    parser.add_argument('--num_workers', type=int, help='Override number of dataloader workers from config')
    parser.add_argument('--fp16_run', action=argparse.BooleanOptionalAction, help='Override FP16/AMP setting from config')
    parser.add_argument('--port', type=int, default=29500, help='Port for distributed training')
    args = parser.parse_args()

    # Load config
    hps = get_hparams_from_file(args.config)

    # --- Override config with command line arguments ---
    # Data paths
    # Prepare training data list and store it in hps
    hps.data.training_files = prepare_datalist(args.data_path)
    if not hps.data.training_files:
         print(f"Error: No valid training files found in '{args.data_path}'. Exiting.")
         return

    # Handle validation files argument
    if args.validation_files is not None:
         hps.data.validation_files = args.validation_files # Store path in hps
    elif not hasattr(hps.data, 'validation_files'):
         hps.data.validation_files = None # Ensure attribute exists, defaults to no validation

    # Output directories
    hps.train.output_dir = args.output_dir # Store base output dir in hps
    # Default checkpoint/log dirs based on output_dir unless overridden
    hps.train.checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join(args.output_dir, "checkpoints")
    hps.train.log_dir = args.log_dir if args.log_dir else os.path.join(args.output_dir, "logs")

    # Checkpoint continuation
    if args.checkpoint_path:
        hps.train.checkpoint_path = args.checkpoint_path
        hps.train.continue_from_checkpoint = True
    else:
        # Check if latest checkpoint exists in default dir if not explicitly continuing
        latest_checkpoint = os.path.join(hps.train.checkpoint_dir, "model_latest.pt")
        if os.path.exists(latest_checkpoint):
             print(f"Found latest checkpoint: {latest_checkpoint}. Resuming automatically.")
             hps.train.checkpoint_path = latest_checkpoint # Point to latest
             hps.train.continue_from_checkpoint = True
        else:
             hps.train.continue_from_checkpoint = False
             hps.train.checkpoint_path = None

    # Training parameters
    if args.epochs is not None: hps.train.epochs = args.epochs
    if args.batch_size is not None: hps.train.batch_size = args.batch_size
    if args.learning_rate is not None: hps.train.learning_rate = args.learning_rate
    if args.seed is not None: hps.train.seed = args.seed
    if args.num_workers is not None: hps.train.num_workers = args.num_workers
    if args.fp16_run is not None: hps.train.fp16_run = args.fp16_run
    #----------------------------------------------------

    # Create output directories (now using potentially updated hps paths)
    os.makedirs(hps.train.output_dir, exist_ok=True)
    os.makedirs(hps.train.checkpoint_dir, exist_ok=True)
    os.makedirs(hps.train.log_dir, exist_ok=True)

    # --- Set random seed for reproducibility ---
    seed = hps.train.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        # Optional: For potentially more deterministic results (can impact performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Determine GPU availability and distributed settings ---
    if torch.cuda.is_available():
        hps.train.num_gpus = torch.cuda.device_count()
        # Enable distributed only if more than 1 GPU is available
        hps.train.distributed = hps.train.num_gpus > 1
        print(f"CUDA available. Found {hps.train.num_gpus} GPU(s). Distributed training: {hps.train.distributed}")
    else:
        hps.train.num_gpus = 0 # Use 0 to indicate no CUDA GPUs
        hps.train.distributed = False
        print("CUDA not available.")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS available (Apple Silicon). Distributed training disabled.")
            # MPS does not support DDP, fp16 might have issues
            if hps.train.fp16_run:
                 print("Warning: FP16 is enabled but may not be fully stable on MPS.")
        else:
            print("Using CPU. Distributed training disabled.")
            if hps.train.fp16_run:
                 print("Warning: FP16 is enabled but has no effect on CPU.")
                 hps.train.fp16_run = False # Disable FP16 for CPU automatically

    # --- Start Training ---
    if hps.train.distributed:
        # Use torch.multiprocessing.spawn for distributed training
        mp.spawn(train, nprocs=hps.train.num_gpus, args=(hps, args.port))
    else:
        # Run directly on rank 0 for single GPU, MPS or CPU
        train(0, hps, args.port)

if __name__ == "__main__":
    main()