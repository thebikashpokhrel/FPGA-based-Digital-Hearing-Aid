"""
utils.py — Audio utilities: STFT, iSTFT, mixing, normalization
"""

import torch
# import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path


# ──────────────────────────────────────────────
#  Audio I/O
# ──────────────────────────────────────────────

AUDIO_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3'}


def load_audio(path: str, target_sr: int = 16000, target_len: int = 16000) -> torch.Tensor:
    """
    Load audio file → mono float32 tensor of fixed length [target_len].
    Uses soundfile instead of torchaudio to avoid FFmpeg dependency.
    """
    audio, sr = sf.read(path, always_2d=False)

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = torch.from_numpy(audio.astype('float32'))

    # Resample if needed
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        audio_np = resample_poly(audio.numpy(), target_sr // g, sr // g)
        audio = torch.from_numpy(audio_np.astype('float32'))

    # Trim or pad to exactly target_len
    if audio.shape[0] > target_len:
        audio = audio[:target_len]
    elif audio.shape[0] < target_len:
        audio = torch.nn.functional.pad(audio, (0, target_len - audio.shape[0]))

    return audio  # [T]


def get_audio_files(directory: str) -> list:
    p = Path(directory)
    return sorted([str(f) for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS])


def save_audio(path: str, waveform: torch.Tensor, sr: int = 16000):
    """Save float32 tensor to WAV using soundfile."""
    audio_np = waveform.squeeze().cpu().numpy()
    sf.write(path, audio_np, sr)


# ──────────────────────────────────────────────
#  SNR-controlled Mixing
# ──────────────────────────────────────────────

def mix_at_snr(voice: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Mix voice + noise at a target SNR (dB).
    Both inputs should be 1-D tensors of same length.

    SNR = 10 * log10(P_voice / P_noise_scaled)
    → scale = sqrt(P_voice / (10^(SNR/10) * P_noise))
    """
    p_voice = voice.pow(2).mean()
    p_noise = noise.pow(2).mean()

    if p_noise < 1e-10:
        return voice

    target_p_noise = p_voice / (10 ** (snr_db / 10.0))
    scale = (target_p_noise / (p_noise + 1e-10)).sqrt()

    mixed = voice + scale * noise

    # Prevent hard clipping
    peak = mixed.abs().max()
    if peak > 1.0:
        mixed = mixed / peak

    return mixed


def random_snr_mix(voice: torch.Tensor, noise: torch.Tensor,
                   snr_min: float = -5.0, snr_max: float = 20.0) -> tuple:
    """
    Randomly pick an SNR in [snr_min, snr_max] and mix.
    Returns (mixed, snr_used).
    """
    snr = float(torch.empty(1).uniform_(snr_min, snr_max).item())
    mixed = mix_at_snr(voice, noise, snr)
    return mixed, snr


# ──────────────────────────────────────────────
#  STFT / iSTFT
# ──────────────────────────────────────────────

class STFTProcessor:
    """
    Wraps torch STFT/iSTFT so the rest of the code stays clean.

    Returns:
        magnitude : [freq_bins, time_frames]   (real, ≥0)
        phase     : [freq_bins, time_frames]   (radians)
        complex   : [freq_bins, time_frames, 2] (real, imag)
    """

    def __init__(self, n_fft=512, hop_length=128, win_length=512,
                 window='hann', center=True, device='cpu'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.device = device
        self.freq_bins = n_fft // 2 + 1   # 257 for n_fft=512

        if window == 'hann':
            self.window = torch.hann_window(win_length).to(device)
        else:
            self.window = torch.ones(win_length).to(device)

    def to(self, device):
        self.device = device
        self.window = self.window.to(device)
        return self

    def transform(self, waveform: torch.Tensor):
        """
        waveform: [T] or [B, T]
        Returns magnitude [(..., F, T)], phase [(..., F, T)]
        """
        batched = waveform.dim() == 2
        if not batched:
            waveform = waveform.unsqueeze(0)   # [1, T]

        B, T = waveform.shape
        specs = []
        for i in range(B):
            s = torch.stft(
                waveform[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                return_complex=True
            )   # [F, T_frames]
            specs.append(s)

        spec = torch.stack(specs, dim=0)   # [B, F, T_frames]

        magnitude = spec.abs()             # [B, F, T_frames]
        phase = spec.angle()               # [B, F, T_frames]

        if not batched:
            magnitude = magnitude.squeeze(0)
            phase = phase.squeeze(0)

        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct waveform from magnitude and phase.
        magnitude, phase: [F, T] or [B, F, T]
        Returns waveform [T] or [B, T]
        """
        batched = magnitude.dim() == 3
        if not batched:
            magnitude = magnitude.unsqueeze(0)
            phase = phase.unsqueeze(0)

        complex_spec = magnitude * torch.exp(1j * phase)

        waveforms = []
        for i in range(complex_spec.shape[0]):
            w = torch.istft(
                complex_spec[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center
            )
            waveforms.append(w)

        waveform = torch.stack(waveforms, dim=0)   # [B, T]

        if not batched:
            waveform = waveform.squeeze(0)

        return waveform


# ──────────────────────────────────────────────
#  Normalization (per-sample, prevents scale issues)
# ──────────────────────────────────────────────

def normalize_waveform(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit RMS normalization."""
    x = x - x.mean()
    rms = x.pow(2).mean().sqrt()
    return x / (rms + eps)


def log_magnitude(magnitude: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Log-magnitude compression: log(1 + magnitude).
    Reduces dynamic range, helps training stability.
    """
    return torch.log1p(magnitude)


def inverse_log_magnitude(log_mag: torch.Tensor) -> torch.Tensor:
    return torch.expm1(log_mag.clamp(min=0))


# ──────────────────────────────────────────────
#  Metrics helpers (numpy-based for eval)
# ──────────────────────────────────────────────

def si_snr_numpy(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.
    Higher is better. Good speech enhancement models reach 15–20 dB.
    """
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()

    # Project estimate onto reference
    dot = np.dot(reference, estimate)
    ref_power = np.dot(reference, reference) + eps
    s_target = (dot / ref_power) * reference

    e_noise = estimate - s_target
    si_snr = 10 * np.log10(
        (np.dot(s_target, s_target) + eps) /
        (np.dot(e_noise, e_noise) + eps)
    )
    return float(si_snr)


def si_snr_torch(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batched SI-SNR loss (negative, for minimization). [B, T] inputs."""
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate  = estimate  - estimate.mean(dim=-1, keepdim=True)

    dot = (reference * estimate).sum(dim=-1, keepdim=True)
    ref_power = reference.pow(2).sum(dim=-1, keepdim=True) + eps
    s_target = (dot / ref_power) * reference

    e_noise = estimate - s_target
    si_snr = 10 * torch.log10(
        (s_target.pow(2).sum(dim=-1) + eps) /
        (e_noise.pow(2).sum(dim=-1) + eps)
    )
    return si_snr   # [B]  (higher = better)
