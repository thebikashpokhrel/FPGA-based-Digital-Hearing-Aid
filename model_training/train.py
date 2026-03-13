"""
train.py — Full training loop for HearSmart speech enhancement

Run:
    python train.py --config configs/config.yaml

Features:
  - Dynamic SNR mixing every epoch (model never sees same mix twice)
  - Combined SI-SNR + MSE loss
  - Cosine LR scheduler with warm restarts
  - Early stopping on val SI-SNR
  - TensorBoard logging
  - Checkpointing (best model + periodic)
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import build_dataloaders
from src.model   import build_model
from src.losses  import build_loss
from src.utils   import STFTProcessor, si_snr_torch


# ──────────────────────────────────────────────
#  Training step
# ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, stft, device, grad_clip, epoch):
    model.train()
    total_loss = total_mse = total_sisnr = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f'Epoch {epoch:03d} [Train]', leave=False)
    for batch_idx, (noisy_wav, clean_wav, snr) in enumerate(pbar):
        noisy_wav = noisy_wav.to(device)   # [B, T]
        clean_wav = clean_wav.to(device)   # [B, T]

        # ── STFT ──────────────────────────────────────────────
        noisy_mag, noisy_phase = stft.transform(noisy_wav)   # [B, F, T_frames]
        clean_mag, _           = stft.transform(clean_wav)

        # ── Forward ───────────────────────────────────────────
        noisy_log_mag  = torch.log1p(noisy_mag)
        mask           = model(noisy_log_mag)          # [B, F, T_frames]
        enhanced_mag   = mask * noisy_mag

        # ── Reconstruct waveform (for SI-SNR loss) ────────────
        enhanced_wav = stft.inverse(enhanced_mag, noisy_phase)   # [B, T]
        # Trim to original length (STFT adds padding)
        T = clean_wav.shape[-1]
        enhanced_wav = enhanced_wav[..., :T]

        # ── Loss ──────────────────────────────────────────────
        loss, l_mse, l_sisnr = criterion(enhanced_mag, clean_mag, enhanced_wav, clean_wav)

        # ── Backward ──────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss   += loss.item()
        total_mse    += l_mse.item()
        total_sisnr  += l_sisnr.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'SI-SNR': f'{l_sisnr.item():.2f}dB'
        })

    n = n_batches
    return total_loss/n, total_mse/n, total_sisnr/n


# ──────────────────────────────────────────────
#  Validation step
# ──────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, stft, device):
    model.eval()
    total_loss = total_mse = total_sisnr = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc='           [Val]  ', leave=False)
    for noisy_wav, clean_wav, snr in pbar:
        noisy_wav = noisy_wav.to(device)
        clean_wav = clean_wav.to(device)

        noisy_mag, noisy_phase = stft.transform(noisy_wav)
        clean_mag, _           = stft.transform(clean_wav)

        noisy_log_mag = torch.log1p(noisy_mag)
        mask          = model(noisy_log_mag)
        enhanced_mag  = mask * noisy_mag

        enhanced_wav = stft.inverse(enhanced_mag, noisy_phase)
        T = clean_wav.shape[-1]
        enhanced_wav = enhanced_wav[..., :T]

        loss, l_mse, l_sisnr = criterion(enhanced_mag, clean_mag, enhanced_wav, clean_wav)

        total_loss  += loss.item()
        total_mse   += l_mse.item()
        total_sisnr += l_sisnr.item()

        pbar.set_postfix({'val_SI-SNR': f'{l_sisnr.item():.2f}dB'})

    n = n_batches
    return total_loss/n, total_mse/n, total_sisnr/n


# ──────────────────────────────────────────────
#  Main training script
# ──────────────────────────────────────────────

def train(config_path: str):
    # ── Load config ───────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Device ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f"  HearSmart Speech Enhancement Training")
    print(f"  Device  : {device}")
    print(f"  Config  : {config_path}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────
    model = build_model(cfg).to(device)

    # ── STFT ──────────────────────────────────────────────────
    s_cfg = cfg['stft']
    stft = STFTProcessor(
        n_fft=s_cfg['n_fft'],
        hop_length=s_cfg['hop_length'],
        win_length=s_cfg['win_length'],
        window=s_cfg['window'],
        center=s_cfg['center'],
        device=device
    )

    # ── Loss & optimizer ──────────────────────────────────────
    criterion = build_loss(cfg)
    t_cfg = cfg['training']

    optimizer = optim.AdamW(
        model.parameters(),
        lr=t_cfg['learning_rate'],
        weight_decay=t_cfg['weight_decay']
    )

    if t_cfg['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_cfg['epochs'],
            eta_min=t_cfg['lr_min']
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )

    # ── Logging ───────────────────────────────────────────────
    log_dir = cfg['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    ckpt_dir = t_cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────
    best_val_sisnr = -float('inf')
    patience_counter = 0
    patience = t_cfg['patience']

    print(f"Starting training for {t_cfg['epochs']} epochs...\n")

    for epoch in range(1, t_cfg['epochs'] + 1):
        t0 = time.time()

        tr_loss, tr_mse, tr_sisnr = train_epoch(
            model, train_loader, optimizer, criterion, stft, device,
            t_cfg['grad_clip'], epoch
        )
        va_loss, va_mse, va_sisnr = val_epoch(
            model, val_loader, criterion, stft, device
        )

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        # ── Scheduler step ────────────────────────────────────
        if t_cfg['lr_scheduler'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(va_sisnr)

        # ── Log ───────────────────────────────────────────────
        writer.add_scalar('Loss/train',      tr_loss,   epoch)
        writer.add_scalar('Loss/val',        va_loss,   epoch)
        writer.add_scalar('SI-SNR/train',    tr_sisnr,  epoch)
        writer.add_scalar('SI-SNR/val',      va_sisnr,  epoch)
        writer.add_scalar('MSE/train',       tr_mse,    epoch)
        writer.add_scalar('MSE/val',         va_mse,    epoch)
        writer.add_scalar('LR',              current_lr, epoch)

        print(
            f"Epoch {epoch:03d}/{t_cfg['epochs']} | "
            f"Train loss={tr_loss:.4f}  SI-SNR={tr_sisnr:.2f}dB | "
            f"Val loss={va_loss:.4f}  SI-SNR={va_sisnr:.2f}dB | "
            f"LR={current_lr:.6f} | {elapsed:.1f}s"
        )

        # ── Save best checkpoint ───────────────────────────────
        if va_sisnr > best_val_sisnr:
            best_val_sisnr = va_sisnr
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_si_snr': va_sisnr,
                'config': cfg,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  ✅ New best model saved (val SI-SNR={va_sisnr:.2f}dB)")
        else:
            patience_counter += 1

        # ── Periodic checkpoint ───────────────────────────────
        if epoch % t_cfg['save_every_n_epochs'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_si_snr': va_sisnr,
            }, os.path.join(ckpt_dir, f'checkpoint_epoch{epoch:03d}.pt'))

        # ── Early stopping ────────────────────────────────────
        if patience_counter >= patience:
            print(f"\n⏹  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\n{'='*55}")
    print(f"  Training complete. Best val SI-SNR: {best_val_sisnr:.2f} dB")
    print(f"  Best model saved to: {ckpt_dir}/best_model.pt")
    print(f"{'='*55}\n")

    writer.close()

    # ── Final test evaluation ─────────────────────────────────
    print("Running final evaluation on test set...")
    from src.evaluate import evaluate_loader
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    evaluate_loader(model, test_loader, stft, device, cfg['evaluation']['results_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    train(args.config)
