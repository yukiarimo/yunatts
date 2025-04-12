import os
import argparse
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
import random
from hanasuconverter import commons
from hanasuconverter.commons import kl_divergence
from hanasuconverter.models import SynthesizerTrn
from hanasuconverter.audio import mel_spectrogram, load_wav_to_torch
from hanasuconverter.utils import (
    get_hparams_from_file,
    save_checkpoint,
    load_checkpoint,
    scan_checkpoint,
)
import torch.amp

class VoiceDataset(torch.utils.data.Dataset):
    """
    Dataset for voice conversion training
    Loads audio, converts to mel spectrogram, returns random segment.
    """
    def __init__(self, data_list, hparams):
        self.data_list = data_list  # List of (audio_path, speaker_name)
        self.hps = hparams
        self.segment_size = hparams.train.segment_size  # Waveform segment size
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

        # Group audio files by speaker for more efficient training
        self.speaker_to_files = {speaker: [] for speaker in self.speakers}
        for audio_path, speaker in data_list:
            self.speaker_to_files[speaker].append(audio_path)

    def __getitem__(self, index):
        # Get audio file and speaker
        audio_path, speaker_name = self.data_list[index]
        speaker_id = self.speaker_to_id[speaker_name]

        # Load audio (target sample rate)
        audio, sr = load_wav_to_torch(audio_path, self.sampling_rate)

        # Handle stereo audio (keep stereo for model input)
        is_stereo = (audio.dim() > 1 and audio.shape[0] > 1)

        # Create mono version for mel calculation if needed
        if is_stereo:
            audio_mono = torch.mean(audio, dim=0)
        else:
            # If already mono, ensure it has shape [1, T]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio_mono = audio[0]

        # Ensure audio is long enough for segment
        if audio_mono.size(0) < self.segment_size:
            # Pad if too short
            pad_size = self.segment_size - audio_mono.size(0)
            audio_mono = F.pad(audio_mono, (0, pad_size), 'constant', 0)
            # Pad original audio (could be stereo)
            if is_stereo:
                audio = F.pad(audio, (0, pad_size), 'constant', 0)
            else:
                audio = F.pad(audio, (0, pad_size), 'constant', 0)
        elif audio_mono.size(0) > self.segment_size:
            # Random segment for training
            max_start = audio_mono.size(0) - self.segment_size
            start = torch.randint(0, max_start + 1, (1,)).item()
            # Slice mono version for mel
            audio_mono = audio_mono[start:start + self.segment_size]
            # Slice original audio (could be stereo)
            if is_stereo:
                audio = audio[:, start:start + self.segment_size]
            else:
                audio = audio[:, start:start + self.segment_size]

        # Generate mel spectrogram from the mono audio
        mel = mel_spectrogram(
            audio_mono.unsqueeze(0),
            self.filter_length,
            self.n_mel_channels,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            self.mel_fmin,
            self.mel_fmax
        ).squeeze(0)  # Remove batch dim, result shape [n_mels, T_mel]

        # Return waveform segment, mel segment, speaker_id
        return audio, mel, speaker_id

    def __len__(self):
        return len(self.data_list)

    def get_random_speaker(self, exclude_speaker=None):
        """Get a random speaker ID different from exclude_speaker"""
        available_speakers = [s for s in self.speakers if s != exclude_speaker]
        if not available_speakers:
            return None  # No other speakers available
        random_speaker = random.choice(available_speakers)
        return self.speaker_to_id[random_speaker]

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

    optimizer.zero_grad(set_to_none=True)  # More efficient zeroing

    # Calculate segment indices for loss
    mel_segment_size = hps.train.segment_size // hps.data.hop_length
    wav_segment_size = hps.train.segment_size

    # Ensure lengths are sufficient for slicing
    min_len = torch.min(y_lengths)
    if min_len < mel_segment_size:
        print(f"Warning: Min y_length ({min_len}) < mel_segment_size ({mel_segment_size}). Required wav length {wav_segment_size}. Skipping batch.")
        return None  # Skip this batch if any sample is too short

    ids_slice = (torch.rand(y.size(0)).to(y.device) * (y_lengths.float() - mel_segment_size + 1)).long()
    wav_ids_slice = ids_slice * hps.data.hop_length

    # Use autocast for mixed precision
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if hps.train.fp16_run and device.type in ['cuda', 'mps'] else torch.float32, enabled=hps.train.fp16_run):
        # Model forward pass
        y_hat, _, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model(
            x, x_lengths, y, y_lengths, sid_src=sid_src, sid_tgt=sid_tgt
        )

        # Slice target mel spectrogram
        y_mel = commons.slice_segments(y, ids_slice, mel_segment_size)

        # Slice predicted waveform
        y_hat_slice = commons.slice_segments(y_hat, wav_ids_slice, wav_segment_size)

        # Convert sliced predicted waveform to mel spectrogram
        y_hat_mono_slice = torch.mean(y_hat_slice, dim=1)  # Convert stereo output to mono for mel loss
        y_hat_mel = mel_spectrogram(
            y_hat_mono_slice,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )

        # Reconstruction loss (L1 on mel spectrograms)
        mel_loss = F.l1_loss(y_hat_mel, y_mel) * hps.train.c_mel

        # KL divergence loss
        kl_loss = kl_divergence(m_q, logs_q, m_p, logs_p)
        kl_loss = torch.sum(kl_loss * z_mask) / torch.sum(z_mask) * hps.train.c_kl

        # Total loss
        total_loss = mel_loss + kl_loss

    # Backward pass using scaler
    scaler.scale(total_loss).backward()

    # Gradient clipping
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.train.grad_clip)

    # Update weights using scaler
    scaler.step(optimizer)
    scaler.update()

    # Return metrics
    grad_norm_item = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
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
    else:
        # Prioritize MPS on Mac, then CUDA, then CPU
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
            print("Using CUDA")
        else:
            device = torch.device('cpu')
            print("Using CPU")

    print(f"Rank {rank} using device: {device}")
    torch.manual_seed(hps.train.seed + rank)  # Ensure different seeds per process if distributed

    # Create dataset BEFORE model to get n_speakers
    if not hasattr(hps.data, 'training_files') or not hps.data.training_files:
        if rank == 0:
            print("Error: Training file list is empty.")
        return

    train_dataset = VoiceDataset(hps.data.training_files, hps)
    n_speakers = len(train_dataset.speakers)  # Get number of speakers
    if rank == 0:
        print(f"Determined {n_speakers} speakers from dataset.")

    # Create model (passing n_speakers)
    model = SynthesizerTrn(
        hps.data.n_mel_channels,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        n_speakers=n_speakers,
        gin_channels=hps.model.gin_channels,
        use_external_speaker_embedding=False
    ).to(device)

    # Wrap model for distributed training if needed
    if hps.train.distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Create optimizer with reduced weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
        weight_decay=hps.train.weight_decay
    )

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=hps.train.fp16_run and device.type in ['cuda', 'mps'])

    # Load checkpoint if continuing training
    global_step = 0
    start_epoch = 0
    if hps.train.continue_from_checkpoint:
        checkpoint_path = hps.train.checkpoint_path
        # Use scan_checkpoint to find the latest if path is a directory
        if os.path.isdir(checkpoint_path):
            actual_checkpoint_path = scan_checkpoint(checkpoint_path, 'model_')
            if not actual_checkpoint_path:
                print(f"Warning: Checkpoint directory specified ({checkpoint_path}), but no checkpoints found. Starting fresh.")
                actual_checkpoint_path = None
            else:
                print(f"Found latest checkpoint in directory: {actual_checkpoint_path}")
        elif os.path.isfile(checkpoint_path):
            actual_checkpoint_path = checkpoint_path  # Use specified file
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
                print(f"Loaded checkpoint at iteration {global_step}")
            except Exception as e:
                print(f"Error loading checkpoint {actual_checkpoint_path}: {e}. Starting fresh.")
                global_step = 0  # Reset step if loading failed
    else:
        print("Starting training from scratch (no checkpoint specified).")

    # Create dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=hps.train.num_gpus,
        rank=rank,
        shuffle=True
    ) if hps.train.distributed else None

    # Use larger batch size and drop_last=True for consistent batch sizes
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=hps.train.num_workers,
        sampler=train_sampler,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    # Calculate start epoch
    if global_step > 0 and len(train_loader) > 0:
        start_epoch = global_step // len(train_loader)
    else:
        start_epoch = 0

    if rank == 0:
        print(f"Starting training from Epoch {start_epoch + 1}")

    # Create learning rate scheduler with warmup
    def lr_lambda(step):
        # Warmup for 1000 steps
        warmup_steps = 1000
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return hps.train.lr_decay ** (step - warmup_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=global_step - 1
    )

    # Create tensorboard writer
    if rank == 0:
        os.makedirs(hps.train.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=hps.train.log_dir)

    # Create checkpoint directory
    if rank == 0:
        os.makedirs(os.path.join(hps.train.output_dir, hps.train.checkpoint_dir), exist_ok=True)

    # Training loop
    model.train()
    for epoch in range(start_epoch, hps.train.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"--- Starting Epoch {epoch + 1}/{hps.train.epochs} ---")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hps.train.epochs}", disable=(rank != 0))

        # Initialize metrics for this epoch
        epoch_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0

        for batch_idx, (x, y, speaker_id) in enumerate(pbar):
            # x: [B, C, T_wav], y: [B, n_mels, T_mel], speaker_id: [B]

            # Calculate lengths
            x_lengths = torch.LongTensor([x.size(-1)] * x.size(0))
            y_lengths = torch.LongTensor([y.size(-1)] * y.size(0))

            # For each sample in the batch, randomly decide if we use the same speaker
            # or a different speaker for source and target
            sid_src = speaker_id.clone()
            sid_tgt = speaker_id.clone()

            # For each sample, with probability (1 - same_speaker_prob),
            # replace sid_src with a different speaker
            for i in range(len(speaker_id)):
                if random.random() > hps.train.same_speaker_prob:
                    # Get a random speaker different from the current one
                    different_speaker = train_dataset.get_random_speaker(exclude_speaker=train_dataset.speakers[speaker_id[i].item()])
                    if different_speaker is not None:
                        sid_src[i] = different_speaker

            # Train step
            metrics = train_step(model, optimizer, scaler, x, x_lengths, y, y_lengths, sid_src, sid_tgt, hps)

            if metrics is not None:
                # Update metrics
                epoch_loss += metrics['total_loss']
                epoch_mel_loss += metrics['mel_loss']
                epoch_kl_loss += metrics['kl_loss']
                epoch_grad_norm += metrics['grad_norm']
                num_batches += 1

                # Update learning rate
                scheduler.step()

                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'total_loss': metrics['total_loss'],
                    'mel_loss': metrics['mel_loss'],
                    'kl_loss': metrics['kl_loss'],
                    'grad_norm': metrics['grad_norm'],
                    'lr': f"{current_lr:.2e}",
                    'step': global_step
                })

                # Log to tensorboard
                if rank == 0 and global_step % hps.train.log_interval == 0:
                    writer.add_scalar('train/loss', metrics['total_loss'], global_step)
                    writer.add_scalar('train/mel_loss', metrics['mel_loss'], global_step)
                    writer.add_scalar('train/kl_loss', metrics['kl_loss'], global_step)
                    writer.add_scalar('train/grad_norm', metrics['grad_norm'], global_step)
                    writer.add_scalar('train/lr', current_lr, global_step)

                # Save checkpoint
                if rank == 0 and global_step % hps.train.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(
                        hps.train.output_dir,
                        hps.train.checkpoint_dir,
                        f"model_{global_step:08d}.pt"
                    )
                    print(f"Saving checkpoint to {checkpoint_path} at iteration {global_step}...")
                    save_checkpoint(
                        model.module if hps.train.distributed else model,
                        optimizer,
                        hps,
                        global_step,
                        checkpoint_path,
                        scaler=scaler
                    )
                    print("Saved checkpoint successfully.")

                    # Save latest checkpoint
                    latest_path = os.path.join(
                        hps.train.output_dir,
                        hps.train.checkpoint_dir,
                        "model_latest.pt"
                    )
                    print(f"Saving checkpoint to {latest_path} at iteration {global_step}...")
                    save_checkpoint(
                        model.module if hps.train.distributed else model,
                        optimizer,
                        hps,
                        global_step,
                        latest_path,
                        scaler=scaler
                    )
                    print("Saved checkpoint successfully.")

                global_step += 1

        # Epoch summary
        if rank == 0 and num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_mel_loss = epoch_mel_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            avg_grad_norm = epoch_grad_norm / num_batches

            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Average Mel Loss: {avg_mel_loss:.4f}")
            print(f"  Average KL Loss: {avg_kl_loss:.4f}")
            print(f"  Average Grad Norm: {avg_grad_norm:.4f}")

            writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            writer.add_scalar('train/epoch_mel_loss', avg_mel_loss, epoch)
            writer.add_scalar('train/epoch_kl_loss', avg_kl_loss, epoch)
            writer.add_scalar('train/epoch_grad_norm', avg_grad_norm, epoch)

    # Clean up
    if rank == 0:
        writer.close()

def prepare_datalist(csv_path):
    """
    Prepare data list from CSV file
    """
    data_list = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:  # Ensure row has at least 2 elements
                    audio_path, speaker_name = row[0], row[1]
                    data_list.append((audio_path, speaker_name))
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")

    return data_list

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Hanasu Converter - Training Script")

    # General arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save output files')

    # Training arguments
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to checkpoint for continuing training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimization')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Load config
    hps = get_hparams_from_file(args.config)

    # Override config with command line arguments
    if args.checkpoint_path:
        hps.train.continue_from_checkpoint = True
        hps.train.checkpoint_path = args.checkpoint_path

    if args.epochs is not None:
        hps.train.epochs = args.epochs

    if args.batch_size is not None:
        hps.train.batch_size = args.batch_size

    if args.learning_rate is not None:
        hps.train.learning_rate = args.learning_rate

    if args.seed is not None:
        hps.train.seed = args.seed

    # Set output directory
    hps.train.output_dir = args.output_dir

    # Prepare data list
    print(f"Reading data list from CSV: {args.data_path}")
    data_list = prepare_datalist(args.data_path)
    print(f"Prepared data list with {len(data_list)} audio files.")

    # Store data list in hps
    hps.data.training_files = data_list

    # Check for CUDA
    if torch.cuda.is_available():
        print(f"CUDA available. Found {torch.cuda.device_count()} devices.")
        if hps.train.distributed and torch.cuda.device_count() > 1:
            hps.train.num_gpus = torch.cuda.device_count()
            mp.spawn(train, nprocs=hps.train.num_gpus, args=(hps, 29500))
        else:
            print("CUDA available but not using distributed training.")
            hps.train.distributed = False
            train(0, hps, 29500)
    else:
        print("CUDA not available.")
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS available (Apple Silicon). Distributed training disabled.")
            hps.train.distributed = False
            train(0, hps, 29500)
        else:
            print("MPS not available. Using CPU.")
            hps.train.distributed = False
            train(0, hps, 29500)

if __name__ == "__main__":
    main()