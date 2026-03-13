"""
losses.py — Combined loss function for speech enhancement

Two complementary losses:
─────────────────────────────────────────────────────────
1. SI-SNR Loss (waveform domain)
   - Measures signal quality directly in the time domain
   - Scale-invariant: handles volume differences correctly
   - Captures perceptual quality well

2. MSE Loss (spectral magnitude domain)
   - Directly penalizes spectral mask errors
   - Stable and easy to optimize
   - Focuses model on getting the right frequency shape

Combined: L = w1 * L_mse + w2 * (-SI-SNR)
          (weights set in config.yaml)
─────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """
    Combined MSE + SI-SNR loss.

    Takes:
        enhanced_mag : [B, F, T]  predicted enhanced magnitude (linear)
        clean_mag    : [B, F, T]  target clean magnitude (linear)
        enhanced_wav : [B, T]     reconstructed waveform (from iSTFT)
        clean_wav    : [B, T]     target clean waveform

    Returns scalar loss (lower = better).
    """

    def __init__(self, mse_weight: float = 0.5, sisnr_weight: float = 0.5):
        super().__init__()
        self.mse_weight    = mse_weight
        self.sisnr_weight  = sisnr_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        enhanced_mag: torch.Tensor,
        clean_mag:    torch.Tensor,
        enhanced_wav: torch.Tensor,
        clean_wav:    torch.Tensor,
    ) -> tuple:

        # ── Spectral MSE ──────────────────────────────────────
        # On log-magnitude to balance across frequencies
        log_enhanced = torch.log1p(enhanced_mag)
        log_clean    = torch.log1p(clean_mag)
        loss_mse = self.mse(log_enhanced, log_clean)

        # ── SI-SNR (negative, since we minimize) ──────────────
        sisnr = self._si_snr(clean_wav, enhanced_wav)    # [B]
        loss_sisnr = -sisnr.mean()                       # scalar

        # ── Combined ──────────────────────────────────────────
        total = self.mse_weight * loss_mse + self.sisnr_weight * loss_sisnr

        return total, loss_mse.detach(), (-loss_sisnr).detach()

    @staticmethod
    def _si_snr(reference: torch.Tensor, estimate: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
        """SI-SNR in dB. [B, T] → [B]"""
        ref = reference - reference.mean(dim=-1, keepdim=True)
        est = estimate  - estimate.mean(dim=-1, keepdim=True)

        dot       = (ref * est).sum(dim=-1, keepdim=True)
        ref_power = ref.pow(2).sum(dim=-1, keepdim=True) + eps
        s_target  = (dot / ref_power) * ref

        e_noise  = est - s_target
        si_snr   = 10 * torch.log10(
            (s_target.pow(2).sum(dim=-1) + eps) /
            (e_noise.pow(2).sum(dim=-1) + eps)
        )
        return si_snr   # [B]


def build_loss(config: dict) -> CombinedLoss:
    return CombinedLoss(
        mse_weight   = config['training']['loss_mse_weight'],
        sisnr_weight = config['training']['loss_sisnr_weight'],
    )
