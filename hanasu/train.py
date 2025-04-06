import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import logging
torch.autograd.set_detect_anomaly(True)
logging.getLogger("numba").setLevel(logging.WARNING)
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

# Allow TF32 for potential performance gains on NVIDIA GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

torch.backends.cudnn.benchmark = True

# Optional: enable flash and/or math SDPA kernels if PyTorch version supports it
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except:
    pass

global_step = 0

def get_device():
    """
    Get the appropriate device for training
    - CUDA for NVIDIA GPUs
    - MPS for Apple Silicon
    - CPU as fallback
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"

def run():
    hps = utils.get_hparams()
    
    # Set up device
    device = get_device()
    print(f"Using device: {device}")
    
    torch.manual_seed(hps.train.seed)
    global global_step
    
    # Set up logging and tensorboard
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # Set up datasets and dataloaders
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    
    # Simple DataLoader without distributed sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=None,  # Reduced from 4 for better compatibility
    )
    
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Set up model parameters
    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        print("Using noise scaled MAS for VITS2")
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        print("Using duration discriminator for VITS2")
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).to(device)
    else:
        net_dur_disc = None

    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder "
                "to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")

    # Set up audio channels (mono or stereo)
    audio_channels = getattr(hps.data, "audio_channels", 1)  # Default to mono if not specified

    # Initialize models
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        channels=audio_channels,  # Pass audio channels to generator
        **hps.model,
    ).to(device)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, channels=audio_channels).to(device)

    # Initialize optimizers
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None

    # Load checkpoints if available
    try:
        if net_dur_disc is not None:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
        _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            net_g,
            optim_g,
            skip_optimizer=hps.train.skip_optimizer
            if "skip_optimizer" in hps.train
            else True,
        )
        _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
            net_d,
            optim_d,
            skip_optimizer=hps.train.skip_optimizer
            if "skip_optimizer" in hps.train
            else True,
        )
        global_step = int(epoch_str) * len(train_loader)
        if "skip_optimizer" not in hps.train or not hps.train.skip_optimizer:
            optim_g.param_groups[0]["lr"] = g_resume_lr
            optim_d.param_groups[0]["lr"] = d_resume_lr
            if net_dur_disc is not None:
                optim_dur_disc.param_groups[0]["lr"] = dur_resume_lr
    except:
        epoch_str = 1
        global_step = 0

    # Set up gradient scaler for mixed precision training
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(int(epoch_str), hps.train.epochs + 1):
        if epoch > int(epoch_str):
            train_loader.dataset.shuffle_mapping()

        train_and_evaluate(
            epoch,
            hps,
            [net_g, net_d, net_dur_disc],
            [optim_g, optim_d, optim_dur_disc],
            scaler,
            [train_loader, eval_loader],
            logger,
            [writer, writer_eval],
            device,
        )

        # Save checkpoints after each epoch
        if epoch % hps.train.save_every_epoch == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(epoch)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(epoch)),
            )
            if net_dur_disc is not None:
                utils.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "DUR_{}.pth".format(epoch)),
                )

def train_and_evaluate(epoch, hps, nets, optims, scaler, loaders, logger, writers, device):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
        net_dur_disc.train()

    # Training loop
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        # Move data to device
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid, llama_emb = [
            b.to(device) for b in batch
        ]

        with autocast(device_type=device, enabled=hps.train.fp16_run):
            # Generator forward
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (x, logw, logw_),  # Add this line to unpack the extra value
            ) = net_g(x, x_lengths, spec, spec_lengths, sid, llama_emb)

            # Discriminator forward - modifying to handle stereo output from generator
            y_mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            
            # For stereo handling - remove squeeze if output is already correct shape
            if y_hat.size(1) == 2:  # Stereo
                y_hat_for_mel = y_hat
            else:
                y_hat_for_mel = y_hat.squeeze(1)
                
            y_hat_mel = mel_spectrogram_torch(
                y_hat_for_mel,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            # Duration discriminator
            if net_dur_disc is not None:
                # logw is already a 3D tensor with shape [batch_size, 1, seq_length]
                # We need to make sure it stays 3D for the Conv1d layers
                # Check and fix the shape if needed
                if logw.dim() == 2:
                    logw = logw.unsqueeze(1)  # Add channel dimension
                elif logw.dim() == 4:
                    logw = logw.squeeze(2)    # Remove extra dimension
                
                # Now use logw which should be [batch_size, 1, seq_length]
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(x, x_mask, logw, logw)
                with autocast(device_type=device, enabled=False):
                    # Compute loss - real value should be close to 1, generated close to 0
                    loss_dur = torch.sum(torch.pow(1 - y_dur_hat_r, 2))
                    loss_dur_hat = torch.sum(torch.pow(0 - y_dur_hat_g, 2))

            # Discriminator loss
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(device_type=device, enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

        # Discriminator backward and optimize
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        # Duration discriminator backward and optimize
        if net_dur_disc is not None:
            optim_dur_disc.zero_grad()
            scaler.scale(loss_dur).backward()
            scaler.unscale_(optim_dur_disc)
            grad_norm_dur_disc = commons.clip_grad_value_(net_dur_disc.parameters(), None)
            scaler.step(optim_dur_disc)

        with autocast(device_type=device, enabled=hps.train.fp16_run):
            # Generator forward again for adversarial training
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(device_type=device, enabled=False):
                # Generator losses
                # Fix for tensor size mismatch
                min_len = min(y_mel.size(3), y_hat_mel.size(3))
                y_mel_matched = y_mel[:, :, :, :min_len]
                y_hat_mel_matched = y_hat_mel[:, :, :, :min_len]
                loss_mel = F.l1_loss(y_mel_matched, y_hat_mel_matched) * hps.train.c_mel
                
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                if net_dur_disc is not None:
                    loss_gen_all += loss_dur_hat.detach()

        # Generator backward and optimize
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        # Update learning rate
       # lr = optim_g.param_groups[0]["lr"]
       # if "learning_rate" in hps.train:
       #     lr = hps.train.learning_rate
       # lr = utils.get_lr_decay(global_step, lr, hps)
       # optim_g.param_groups[0]["lr"] = lr
       # optim_d.param_groups[0]["lr"] = lr
       # if net_dur_disc is not None:
       #     optim_dur_disc.param_groups[0]["lr"] = lr

        # Log training progress
        if global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
            logger.info(
                "Train Epoch: {} [{:.0f}%]".format(
                    epoch, 100.0 * batch_idx / len(train_loader)
                )
            )
            logger.info([x.item() for x in losses] + [global_step, lr])

            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
            }
            scalar_dict.update(
                {
                    "loss/g/fm": loss_fm,
                    "loss/g/gen": loss_gen,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl,
                }
            )

            scalar_dict.update(
                {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
            )
            scalar_dict.update(
                {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
            )
            scalar_dict.update(
                {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
            )
            is_stereo_train = y.size(1) == 2
            is_stereo_spec = spec.size(1) == 2
            image_dict = {
                "slice/mel_org": utils.plot_spectrogram_to_numpy(
                    y_mel[0, 0].data.cpu().numpy() if is_stereo_train else y_mel[0].data.cpu().numpy()
                ),
                "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                    y_hat_mel[0, 0].data.cpu().numpy() if is_stereo_train else y_hat_mel[0].data.cpu().numpy()
                ),
                "all/mel": utils.plot_spectrogram_to_numpy(
                    spec[0, 0].data.cpu().numpy() if is_stereo_spec else spec[0].data.cpu().numpy()
                ),
                "all/attn": utils.plot_alignment_to_numpy(
                    attn[0, 0].data.cpu().numpy()
                ),
            }

            utils.summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=scalar_dict,
            )

        # Evaluate on validation set
        if global_step % hps.train.eval_interval == 0:
            evaluate(hps, net_g, eval_loader, writer_eval, device)
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
            if net_dur_disc is not None:
                utils.save_checkpoint(
                    net_dur_disc,
                    optim_dur_disc,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                )

        global_step += 1

def evaluate(hps, generator, eval_loader, writer_eval, device):
    generator.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid, llama_emb = [
                b.to(device) for b in batch
            ] # spec shape: [B, C, F, T] or [B, F, T]. y shape: [B, C, T] or [B, 1, T]

            # --- Refined Stereo Detection ---
            # Check dimensions of input tensors from the batch
            is_stereo_input_spec = spec.dim() == 4
            is_stereo_input_wav = y.size(1) > 1 # Check channel dim of waveform
            # Use input spec dim as the primary indicator for consistency
            is_stereo = is_stereo_input_spec
            # --- End Refined Stereo Detection ---

            # Generate output
            # generator.infer(...) -> y_hat shape: [B, C, T] or [B, 1, T]
            y_hat, attn, mask, *_ = generator.infer(
                x, x_lengths, sid, llama_emb, max_len=1000
            )
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            # Convert GT spec to mel
            # spec_to_mel_torch(...) -> mel shape: [B, C, M, T] or [B, M, T]
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            # Generate mel from y_hat
            # mel_spectrogram_torch(...) -> y_hat_mel shape: [B, C, M, T] or [B, M, T]
            y_hat_mel = mel_spectrogram_torch(
                y_hat.float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            # --- Consistent Channel Selection for Visualization ---
            # Select the first batch item (index 0).
            # Select the first channel (index 0) *if* input was stereo.
            mel_vis = mel[0, 0] if is_stereo else mel[0]
            y_hat_mel_vis = y_hat_mel[0, 0] if is_stereo else y_hat_mel[0]
            y_vis = y[0, 0] if is_stereo else y[0, 0] # Select first channel even if mono y is [B, 1, T]
            y_hat_vis = y_hat[0, 0] if is_stereo else y_hat[0, 0] # Select first channel even if mono y_hat is [B, 1, T]
            # --- End Consistent Channel Selection ---

            # --- Plotting ---
            # Add debug prints to check shapes right before plotting
            # print(f"DEBUG Eval[{batch_idx}]: is_stereo={is_stereo}")
            # print(f"DEBUG Eval[{batch_idx}]: y_hat_mel_vis shape: {y_hat_mel_vis.shape}")

            image_dict = {
                f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                    y_hat_mel_vis.cpu().numpy() # Should be 2D [M, T]
                )
            }

            # print(f"DEBUG Eval[{batch_idx}]: y_hat_vis shape: {y_hat_vis.shape}")
            audio_dict = {f"gen/audio_{batch_idx}": y_hat_vis[: y_hat_lengths[0]]} # Use selected channel audio

            if global_step == 0:
                 # print(f"DEBUG Eval[{batch_idx}] GS=0: mel_vis shape: {mel_vis.shape}")
                 image_dict.update(
                     {
                         f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                             mel_vis.cpu().numpy() # Should be 2D [M, T]
                         )
                     }
                 )
                 # print(f"DEBUG Eval[{batch_idx}] GS=0: y_vis shape: {y_vis.shape}")
                 audio_dict.update({f"gt/audio_{batch_idx}": y_vis[: y_lengths[0]]}) # Use selected channel audio

            utils.summarize(
                writer=writer_eval,
                global_step=global_step,
                images=image_dict,
                audios=audio_dict,
                audio_sampling_rate=hps.data.sampling_rate,
            )

            if batch_idx >= hps.train.eval_num_samples:
                break

    generator.train()

if __name__ == "__main__":
    run()
