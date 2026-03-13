# HearSmart — Speech Enhancement Neural Network

Conv1D spectral masking network for FPGA-targeted speech enhancement.
Part of the HearSmart adaptive hearing aid project, Pulchowk Campus.

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> If you don't have a GPU, PyTorch will fall back to CPU automatically.
> Training will be slower (~5–10× ) but still works.

---

### 2. Organize your data

```
hearsmart_enhancement/
└── data/
    ├── clean_speech/     ← your 5000 clean 1-second WAV files
    ├── noise/            ← your 5000 noise 1-second WAV files
    └── noisy_mixed/      ← your 5000 pre-mixed files (optional)
```

The script will automatically:
- Split clean + noise files into 80% train / 10% val / 10% test
- Dynamically mix clean + noise at random SNR every training step
- Use fixed pairings for val/test so metrics are comparable epoch-to-epoch

---

### 3. Edit config (optional)

Open `configs/config.yaml` and set:
```yaml
data:
  clean_dir: ./data/clean_speech     # ← update if different
  noise_dir: ./data/noise
  snr_min: -5.0
  snr_max: 20.0
```

Default settings are already tuned for your dataset size (5000 files each).

---

### 4. Train

```bash
python train.py --config configs/config.yaml
```

**What you'll see:**
```
Epoch 001/100 | Train loss=0.4821  SI-SNR=4.21dB | Val loss=0.4103  SI-SNR=6.87dB | LR=0.001000
  ✅ New best model saved (val SI-SNR=6.87dB)
Epoch 002/100 | Train loss=0.3912  SI-SNR=7.14dB | Val loss=0.3641  SI-SNR=8.23dB | LR=0.000999
  ✅ New best model saved (val SI-SNR=8.23dB)
...
```

Training saves:
- `checkpoints/best_model.pt` — best checkpoint by val SI-SNR
- `checkpoints/checkpoint_epochXXX.pt` — periodic checkpoints every 5 epochs
- `logs/` — TensorBoard logs

---

### 5. Monitor training (TensorBoard)

```bash
tensorboard --logdir logs/
```
Then open `http://localhost:6006` in your browser.
You'll see loss curves, SI-SNR curves, and learning rate over epochs.

---

### 6. Evaluate

```bash
python -m src.evaluate \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --output_dir results/
```

**Output:**
```
=======================================================
  EVALUATION RESULTS
=======================================================
  Samples evaluated    : 500
  SI-SNR (noisy)       : 3.21 dB  (baseline)
  SI-SNR (enhanced)    : 14.87 dB
  SI-SNR improvement   : +11.66 dB  ← key metric
  PESQ                 : 3.24  (range -0.5 to 4.5)
  STOI                 : 0.91  (range 0 to 1)
=======================================================

  Interpretation guide:
  SI-SNR improvement > 5dB  → meaningful enhancement
  SI-SNR improvement > 10dB → excellent enhancement
  PESQ > 3.0                → good perceptual quality
  STOI > 0.85               → high intelligibility
```

Also saves side-by-side audio files in `results/`:
- `sample0001_noisy.wav`
- `sample0001_enhanced.wav`
- `sample0001_clean.wav`

Listen to these to judge real perceptual quality.

---

## Understanding the data flow

```
clean_speech/*.wav ─┐
                    ├─→ random SNR mix ─→ noisy_wav  ─┐
noise/*.wav ────────┘                                  │
                                                       ▼
                                          STFT → noisy_magnitude + phase
                                                       │
                                                   log1p()
                                                       │
                                               Conv1D network
                                                       │
                                               soft mask [0,1]
                                                       │
                                     enhanced_mag = mask × noisy_mag
                                                       │
                                          iSTFT (using noisy phase)
                                                       │
                                              enhanced_wav
                                                       │
                                          SI-SNR + MSE loss vs clean
```

---

## What good training looks like

| Epoch | Val SI-SNR | Meaning                          |
|-------|-----------|----------------------------------|
| 1-5   | 3-7 dB    | Model is learning basic patterns |
| 10-20 | 8-12 dB   | Good noise suppression starting  |
| 30+   | 13-17 dB  | Strong enhancement               |
| 50+   | 17-20 dB  | Excellent (diminishing returns)  |

If val SI-SNR is not improving after epoch 20, try:
- Reducing learning rate: `lr: 0.0005`
- Increasing model size: `channels: [128, 256, 256, 128]`
- More training data (data augmentation)

---

## File reference

| File               | Purpose                                         |
|--------------------|-------------------------------------------------|
| `src/dataset.py`   | Data loading + dynamic SNR mixing               |
| `src/model.py`     | Conv1D masking network architecture             |
| `src/losses.py`    | Combined SI-SNR + MSE loss                      |
| `src/utils.py`     | STFT, audio I/O, SNR mixing math                |
| `src/evaluate.py`  | PESQ / STOI / SI-SNR evaluation                 |
| `train.py`         | Main training script                            |
| `configs/config.yaml` | All hyperparameters                          |
