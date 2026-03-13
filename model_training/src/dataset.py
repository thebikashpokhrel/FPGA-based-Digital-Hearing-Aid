"""
dataset.py — PyTorch Dataset with dynamic SNR mixing

Three data sources:
  1. clean_speech/  — used as speech targets
  2. noise/         — mixed dynamically with clean at random SNR
  3. noisy_mixed/   — optional; used as held-out validation/test set

Training strategy:
  - 80% of clean + 80% of noise files → train set (dynamic mixing each epoch)
  - 10% of each → validation set
  - 10% of each → test set
  - Pre-mixed files (noisy_mixed/) used as additional fixed validation to track real-world performance
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from .utils import load_audio, get_audio_files, random_snr_mix, normalize_waveform

class AudioAugmenter:
    """
    Applies random augmentations to clean speech before mixing.
    This multiplies effective dataset size without needing more files.
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Apply each augmentation randomly and independently
        if torch.rand(1) < 0.5:
            waveform = self._random_gain(waveform)
        if torch.rand(1) < 0.3:
            waveform = self._pitch_shift_approx(waveform)
        if torch.rand(1) < 0.3:
            waveform = self._time_stretch_approx(waveform, target_len=waveform.shape[0])
        if torch.rand(1) < 0.4:
            waveform = self._add_small_reverb(waveform)
        return waveform

    def _random_gain(self, x: torch.Tensor) -> torch.Tensor:
        """Random gain between -6dB and +6dB"""
        gain = 10 ** (torch.empty(1).uniform_(-6, 6).item() / 20)
        return (x * gain).clamp(-1.0, 1.0)

    def _pitch_shift_approx(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate pitch shift by resampling then trimming/padding"""
        factor = torch.empty(1).uniform_(0.9, 1.1).item()
        new_len = int(len(x) * factor)
        # Resample
        x_np = x.numpy()
        from scipy.signal import resample
        x_resampled = resample(x_np, new_len)
        x_resampled = torch.from_numpy(x_resampled.astype('float32'))
        # Trim or pad back to original length
        orig_len = len(x)
        if len(x_resampled) > orig_len:
            return x_resampled[:orig_len]
        else:
            return torch.nn.functional.pad(x_resampled, (0, orig_len - len(x_resampled)))

    def _time_stretch_approx(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Approximate time stretch without pitch change"""
        factor = torch.empty(1).uniform_(0.85, 1.15).item()
        new_len = int(target_len * factor)
        from scipy.signal import resample
        x_np = x.numpy()
        stretched = resample(x_np, new_len)
        stretched = torch.from_numpy(stretched.astype('float32'))
        if len(stretched) > target_len:
            return stretched[:target_len]
        else:
            return torch.nn.functional.pad(stretched, (0, target_len - len(stretched)))

    def _add_small_reverb(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate small room reverb with a simple FIR"""
        # Simple 3-tap early reflection simulation
        delays = [int(0.01 * self.sr), int(0.025 * self.sr)]  # 10ms, 25ms
        gains  = [0.3, 0.15]
        result = x.clone()
        for d, g in zip(delays, gains):
            padded = torch.nn.functional.pad(x, (d, 0))[:len(x)]
            result = result + g * padded
        # Normalize to prevent clipping
        peak = result.abs().max()
        if peak > 1.0:
            result = result / peak
        return result

class SpeechEnhancementDataset(Dataset):
    """
    Dynamically mixes clean speech + noise at random SNR every epoch.
    This gives the model exposure to a huge variety of SNR conditions,
    far beyond the fixed 5000 pre-mixed files.

    Returns:
        noisy   : [T] float32 tensor
        clean   : [T] float32 tensor
        snr     : float (the SNR used for this mix)
    """

    def __init__(
        self,
        clean_files: list,
        noise_files: list,
        sample_rate: int = 16000,
        duration: float = 1.0,
        snr_min: float = -5.0,
        snr_max: float = 20.0,
        augment: bool = True,       # True for train, False for val/test
        seed: int = None,           # fix seed for reproducible val/test sets
    ):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.sample_rate = sample_rate
        self.augmenter = AudioAugmenter(sample_rate) if augment else None
        self.target_len = int(sample_rate * duration)
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.augment = augment
        self.seed = seed

        # For reproducible val/test: pre-compute noise pairings and SNRs
        if not augment and seed is not None:
            rng = random.Random(seed)
            np.random.seed(seed)
            self._fixed_noise = [rng.choice(noise_files) for _ in clean_files]
            self._fixed_snrs = [
                float(np.random.uniform(snr_min, snr_max))
                for _ in clean_files
            ]

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load clean speech
        clean = load_audio(self.clean_files[idx], self.sample_rate, self.target_len)
        clean = normalize_waveform(clean)
        if self.augmenter is not None:
            clean = self.augmenter(clean)
            clean = normalize_waveform(clean)  # re-normalize after augmentation

        # Pick noise file
        if self.augment:
            # Training: random noise file, random SNR every call
            noise_path = random.choice(self.noise_files)
            noise = load_audio(noise_path, self.sample_rate, self.target_len)
            noise = normalize_waveform(noise)

            if torch.rand(1) < 0.3:
                noise2_path = random.choice(self.noise_files)
                noise2 = load_audio(noise2_path, self.sample_rate, self.target_len)
                noise2 = normalize_waveform(noise2)
                noise = 0.6 * noise + 0.4 * noise2
                noise = normalize_waveform(noise)
                
            noisy, snr = random_snr_mix(clean, noise, self.snr_min, self.snr_max)
        else:
            # Val/test: fixed pairings so metrics are comparable across epochs
            noise_path = self._fixed_noise[idx]
            noise = load_audio(noise_path, self.sample_rate, self.target_len)
            noise = normalize_waveform(noise)
            from .utils import mix_at_snr
            snr = self._fixed_snrs[idx]
            noisy = mix_at_snr(clean, noise, snr)

        return noisy.float(), clean.float(), torch.tensor(snr, dtype=torch.float32)


class FixedMixDataset(Dataset):
    """
    Loads pre-mixed noisy files paired with clean originals.
    Used for a truly held-out real-world test set.

    Expects filenames to encode the clean source, e.g.:
        voice001__noise-noise042__snr+3.7dB__mix1.wav
    and the clean file to exist in clean_dir as voice001.wav

    If pairing by name isn't possible, set paired=False and it
    will just load noisy files and return None for clean.
    """

    def __init__(
        self,
        mixed_dir: str,
        clean_dir: str,
        sample_rate: int = 16000,
        duration: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration)

        mixed_files = get_audio_files(mixed_dir)
        clean_lookup = {
            Path(f).stem: f for f in get_audio_files(clean_dir)
        }

        self.pairs = []
        for mf in mixed_files:
            stem = Path(mf).stem          # e.g. voice001__noise-noise042__snr+3.7dB__mix1
            # The clean name is the part before '__noise-'
            clean_key = stem.split('__noise-')[0] if '__noise-' in stem else None
            clean_path = clean_lookup.get(clean_key)
            self.pairs.append((mf, clean_path))

        paired = sum(1 for _, c in self.pairs if c is not None)
        print(f"FixedMixDataset: {len(self.pairs)} mixed files, {paired} paired with clean")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mixed_path, clean_path = self.pairs[idx]
        noisy = load_audio(mixed_path, self.sample_rate, self.target_len)
        noisy = normalize_waveform(noisy)

        if clean_path:
            clean = load_audio(clean_path, self.sample_rate, self.target_len)
            clean = normalize_waveform(clean)
        else:
            clean = torch.zeros(self.target_len)

        return noisy.float(), clean.float(), torch.tensor(0.0)


# ──────────────────────────────────────────────
#  Split helpers
# ──────────────────────────────────────────────

def split_files(files: list, train=0.8, val=0.1, test=0.1, seed=42):
    """
    Shuffle and split. Uses fixed seed so splits are always identical
    regardless of filesystem ordering.
    """
    assert abs(train + val + test - 1.0) < 1e-6
    # Use numpy for reproducible shuffle (random.Random gives different
    # results depending on Python version)
    rng = np.random.default_rng(seed)
    files = list(files)
    indices = rng.permutation(len(files)).tolist()
    files = [files[i] for i in indices]

    n = len(files)
    n_train = int(n * train)
    n_val   = int(n * val)
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def build_dataloaders(config: dict, num_workers: int = 4):
    """
    Build train, val, test DataLoaders from config dict.

    config keys: clean_dir, noise_dir, mixed_dir, sample_rate,
                 snr_min, snr_max, train_split, val_split, test_split,
                 batch_size, num_workers
    """
    sr       = config['audio']['sample_rate']
    duration = config['audio']['duration']
    snr_min  = config['data']['snr_min']
    snr_max  = config['data']['snr_max']
    bs       = config['training']['batch_size']
    nw       = config['data']['num_workers']

    clean_files = get_audio_files(config['data']['clean_dir'])
    noise_files = get_audio_files(config['data']['noise_dir'])

    if not clean_files:
        raise FileNotFoundError(f"No audio in {config['data']['clean_dir']}")
    if not noise_files:
        raise FileNotFoundError(f"No audio in {config['data']['noise_dir']}")

    print(f"Clean files : {len(clean_files)}")
    print(f"Noise files : {len(noise_files)}")

    # Split clean and noise independently
    c_train, c_val, c_test = split_files(
        clean_files,
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split']
    )
    n_train, n_val, n_test = split_files(
        noise_files,
        config['data']['train_split'],
        config['data']['val_split'],
        config['data']['test_split']
    )

    print(f"\nSplit → Train: {len(c_train)} | Val: {len(c_val)} | Test: {len(c_test)}")
    print(f"  Using fixed seed splits for reproducible val/test evaluation")

    train_ds = SpeechEnhancementDataset(
        c_train, n_train, sr, duration, snr_min, snr_max, augment=True
    )
    val_ds = SpeechEnhancementDataset(
        c_val, n_val, sr, duration, snr_min, snr_max, augment=False, seed=42
    )
    test_ds = SpeechEnhancementDataset(
        c_test, n_test, sr, duration, snr_min, snr_max, augment=False, seed=99
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader
