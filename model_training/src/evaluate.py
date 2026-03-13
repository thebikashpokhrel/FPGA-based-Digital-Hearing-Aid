"""
evaluate.py — Full evaluation pipeline

Metrics computed:
─────────────────────────────────────────────
  SI-SNR  (Scale-Invariant SNR)      dB  ↑ higher better
  PESQ    (Perceptual Eval of Speech) MOS ↑ higher better  range: -0.5 to 4.5
  STOI    (Short-Time Obj Intelligib) 0-1 ↑ higher better
─────────────────────────────────────────────

Run standalone:
    python -m src.evaluate \
        --config configs/config.yaml \
        --checkpoint checkpoints/best_model.pt \
        --output_dir results/
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("⚠️  pesq not installed — PESQ metric will be skipped. Run: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("⚠️  pystoi not installed — STOI metric will be skipped. Run: pip install pystoi")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils   import STFTProcessor, si_snr_numpy
from src.model   import build_model
from src.dataset import build_dataloaders


# ──────────────────────────────────────────────
#  Per-sample evaluation
# ──────────────────────────────────────────────

def evaluate_sample(
    clean_np: np.ndarray,
    enhanced_np: np.ndarray,
    noisy_np: np.ndarray,
    sr: int = 16000,
) -> dict:
    """
    Compute all metrics for a single sample.
    Returns dict with keys: si_snr_enhanced, si_snr_noisy, pesq, stoi
    """
    results = {}

    # SI-SNR (enhanced vs clean)
    results['si_snr_enhanced'] = si_snr_numpy(clean_np, enhanced_np)
    # SI-SNR (noisy vs clean) — baseline to show improvement
    results['si_snr_noisy']    = si_snr_numpy(clean_np, noisy_np)
    results['si_snr_improvement'] = results['si_snr_enhanced'] - results['si_snr_noisy']

    # PESQ (requires 8kHz or 16kHz)
    if PESQ_AVAILABLE:
        try:
            results['pesq'] = float(pesq(sr, clean_np, enhanced_np, 'wb'))  # wideband
        except Exception:
            results['pesq'] = float('nan')
    else:
        results['pesq'] = float('nan')

    # STOI
    if STOI_AVAILABLE:
        try:
            results['stoi'] = float(stoi(clean_np, enhanced_np, sr, extended=False))
        except Exception:
            results['stoi'] = float('nan')
    else:
        results['stoi'] = float('nan')

    return results


# ──────────────────────────────────────────────
#  Evaluate a full DataLoader
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    stft: STFTProcessor,
    device: torch.device,
    output_dir: str = None,
    save_audio_n: int = 10,       # save first N enhanced samples as WAV
    sr: int = 16000,
):
    model.eval()
    os.makedirs(output_dir, exist_ok=True) if output_dir else None

    all_si_snr_enhanced = []
    all_si_snr_noisy    = []
    all_si_snr_improve  = []
    all_pesq            = []
    all_stoi            = []

    saved = 0

    for batch_idx, (noisy_wav, clean_wav, snr) in enumerate(tqdm(loader, desc='Evaluating')):
        noisy_wav = noisy_wav.to(device)   # [B, T]
        clean_wav = clean_wav.to(device)

        # STFT + enhance
        noisy_mag, noisy_phase = stft.transform(noisy_wav)
        noisy_log_mag = torch.log1p(noisy_mag)
        mask = model(noisy_log_mag)
        enhanced_mag = mask * noisy_mag

        # Reconstruct waveform
        enhanced_wav = stft.inverse(enhanced_mag, noisy_phase)
        T = clean_wav.shape[-1]
        enhanced_wav = enhanced_wav[..., :T]

        # Move to CPU numpy for metric computation
        noisy_np    = noisy_wav.cpu().numpy()       # [B, T]
        clean_np    = clean_wav.cpu().numpy()
        enhanced_np = enhanced_wav.cpu().numpy()

        for i in range(clean_np.shape[0]):
            metrics = evaluate_sample(clean_np[i], enhanced_np[i], noisy_np[i], sr)
            all_si_snr_enhanced.append(metrics['si_snr_enhanced'])
            all_si_snr_noisy.append(metrics['si_snr_noisy'])
            all_si_snr_improve.append(metrics['si_snr_improvement'])
            if not np.isnan(metrics['pesq']):
                all_pesq.append(metrics['pesq'])
            if not np.isnan(metrics['stoi']):
                all_stoi.append(metrics['stoi'])

            # Save audio samples
            if output_dir and saved < save_audio_n:
                idx = batch_idx * loader.batch_size + i
                sf.write(os.path.join(output_dir, f'sample{idx:04d}_noisy.wav'),
                         noisy_np[i], sr)
                sf.write(os.path.join(output_dir, f'sample{idx:04d}_enhanced.wav'),
                         enhanced_np[i], sr)
                sf.write(os.path.join(output_dir, f'sample{idx:04d}_clean.wav'),
                         clean_np[i], sr)
                saved += 1

    # ── Summary ───────────────────────────────────────────────
    summary = {
        'n_samples':         len(all_si_snr_enhanced),
        'si_snr_enhanced':   float(np.mean(all_si_snr_enhanced)),
        'si_snr_noisy':      float(np.mean(all_si_snr_noisy)),
        'si_snr_improvement':float(np.mean(all_si_snr_improve)),
        'pesq':              float(np.mean(all_pesq)) if all_pesq else float('nan'),
        'stoi':              float(np.mean(all_stoi)) if all_stoi else float('nan'),
    }

    print("\n" + "="*55)
    print("  EVALUATION RESULTS")
    print("="*55)
    print(f"  Samples evaluated    : {summary['n_samples']}")
    print(f"  SI-SNR (noisy)       : {summary['si_snr_noisy']:.2f} dB  (baseline)")
    print(f"  SI-SNR (enhanced)    : {summary['si_snr_enhanced']:.2f} dB")
    print(f"  SI-SNR improvement   : {summary['si_snr_improvement']:+.2f} dB  ← key metric")
    if not np.isnan(summary['pesq']):
        print(f"  PESQ                 : {summary['pesq']:.3f}  (range -0.5 to 4.5)")
    if not np.isnan(summary['stoi']):
        print(f"  STOI                 : {summary['stoi']:.3f}  (range 0 to 1)")
    print("="*55)

    # What the numbers mean
    print("\n  Interpretation guide:")
    print("  SI-SNR improvement > 5dB  → meaningful enhancement")
    print("  SI-SNR improvement > 10dB → excellent enhancement")
    print("  PESQ > 3.0                → good perceptual quality")
    print("  STOI > 0.85               → high intelligibility\n")

    if output_dir:
        # Save summary to text file
        with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
        print(f"  Results saved to: {output_dir}")

    return summary


# ──────────────────────────────────────────────
#  Standalone entry point
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--output_dir', default='results/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"val SI-SNR={ckpt.get('val_si_snr', 0):.2f}dB")

    s_cfg = cfg['stft']
    stft = STFTProcessor(
        n_fft=s_cfg['n_fft'], hop_length=s_cfg['hop_length'],
        win_length=s_cfg['win_length'], window=s_cfg['window'],
        center=s_cfg['center'], device=device
    )

    _, _, test_loader = build_dataloaders(cfg)
    evaluate_loader(model, test_loader, stft, device, args.output_dir, sr=cfg['audio']['sample_rate'])
