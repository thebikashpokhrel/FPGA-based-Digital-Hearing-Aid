"""
model.py — Conv1D Spectral Masking Network for Speech Enhancement

Architecture overview:
─────────────────────────────────────────────────────────
Input:  log-magnitude spectrum  [B, F, T]   (F=257 bins, T=time frames)
        (treating frequency as "channels" and time as the 1D sequence)

→ Encoder: 3 × Conv1D blocks with increasing channels
→ Bottleneck: context block with larger receptive field
→ Decoder: 3 × Conv1D blocks mirroring encoder (skip connections)
→ Mask head: Conv1D → Sigmoid → mask in [0,1]

Output: soft mask [B, F, T]
        applied as:  enhanced_mag = mask * noisy_mag

Why masking instead of direct prediction?
- More stable training (mask is bounded 0-1)
- Naturally suppresses noise (mask ≈ 0) and preserves speech (mask ≈ 1)
- Preserves noisy phase for reconstruction (phase is hard to predict)

FPGA note:
- No BatchNorm (hard to implement in streaming HLS)
- Uses LayerNorm over frequency axis instead
- All activations are ReLU or Sigmoid (easy to approximate in fixed-point)
- No attention, no transformer blocks (too resource-heavy)
─────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    1D Conv block: Conv → LayerNorm → ReLU → Dropout
    Operates along the time axis, treating frequency bins as channels.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=pad, dilation=dilation, bias=False)
        self.norm = nn.GroupNorm(1, out_ch)   # equivalent to LayerNorm over channels
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


class ResidualConvBlock(nn.Module):
    """
    ConvBlock with residual skip if in_ch == out_ch.
    Helps gradient flow in deeper models.
    """

    def __init__(self, ch: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.block = ConvBlock(ch, ch, kernel_size, dilation, dropout)

    def forward(self, x):
        return x + self.block(x)


class SpeechEnhancementNet(nn.Module):
    """
    Lightweight Conv1D speech enhancement network.

    Input : noisy log-magnitude  [B, F, T]   F = n_fft//2+1 = 257
    Output: soft mask             [B, F, T]   values in (0, 1)

    Usage:
        model = SpeechEnhancementNet(in_bins=257)
        mask  = model(noisy_log_mag)
        enhanced_mag = mask * noisy_mag   # apply to linear magnitude
    """

    # def __init__(
    #     self,
    #     in_bins: int = 257,
    #     channels: list = None,    # [64, 128, 128, 64]
    #     kernel_size: int = 3,
    #     dropout: float = 0.1,
    # ):
    #     super().__init__()

    #     if channels is None:
    #         channels = [64, 128, 128, 64]

    #     # ── Input projection ──────────────────────────────────
    #     # in_bins → channels[0]
    #     self.input_proj = ConvBlock(in_bins, channels[0], kernel_size=1, dropout=0.0)

    #     # ── Encoder ───────────────────────────────────────────
    #     # Stack of conv blocks with increasing dilation for larger receptive field
    #     self.encoder = nn.Sequential(
    #         ConvBlock(channels[0], channels[1], kernel_size, dilation=1, dropout=dropout),
    #         ConvBlock(channels[1], channels[1], kernel_size, dilation=2, dropout=dropout),
    #         ConvBlock(channels[1], channels[2], kernel_size, dilation=4, dropout=dropout),
    #         ConvBlock(channels[2], channels[2], kernel_size, dilation=8, dropout=dropout),
    #     )

    #     # ── Bottleneck (wide context) ──────────────────────────
    #     self.bottleneck = nn.Sequential(
    #         ResidualConvBlock(channels[2], kernel_size=5, dilation=1,  dropout=dropout),
    #         ResidualConvBlock(channels[2], kernel_size=5, dilation=2,  dropout=dropout),
    #         ResidualConvBlock(channels[2], kernel_size=5, dilation=4,  dropout=dropout),
    #     )

    #     # ── Decoder ───────────────────────────────────────────
    #     self.decoder = nn.Sequential(
    #         ConvBlock(channels[2], channels[2], kernel_size, dilation=4, dropout=dropout),
    #         ConvBlock(channels[2], channels[1], kernel_size, dilation=2, dropout=dropout),
    #         ConvBlock(channels[1], channels[1], kernel_size, dilation=1, dropout=dropout),
    #         ConvBlock(channels[1], channels[0], kernel_size, dilation=1, dropout=dropout),
    #     )

    #     # ── Mask output ───────────────────────────────────────
    #     # Linear conv then sigmoid to get mask in (0, 1)
    #     self.mask_head = nn.Sequential(
    #         nn.Conv1d(channels[0], in_bins, kernel_size=1),
    #         nn.Sigmoid()
    #     )

    #     self._init_weights()
    def __init__(self, in_bins=257, channels=None, kernel_size=3, dropout=0.05):
        super().__init__()
        if channels is None:
            channels = [128, 256, 256, 128]

        self.input_proj = ConvBlock(in_bins, channels[0], kernel_size=1, dropout=0.0)

        self.encoder = nn.Sequential(
            ConvBlock(channels[0], channels[1], kernel_size, dilation=1, dropout=dropout),
            ConvBlock(channels[1], channels[1], kernel_size, dilation=2, dropout=dropout),
            ConvBlock(channels[1], channels[2], kernel_size, dilation=4, dropout=dropout),
            ConvBlock(channels[2], channels[2], kernel_size, dilation=8, dropout=dropout),
        )

        self.bottleneck = nn.Sequential(
            ResidualConvBlock(channels[2], kernel_size=5, dilation=1, dropout=dropout),
            ResidualConvBlock(channels[2], kernel_size=5, dilation=2, dropout=dropout),
            ResidualConvBlock(channels[2], kernel_size=5, dilation=4, dropout=dropout),
        )

        self.decoder = nn.Sequential(
            ConvBlock(channels[2], channels[2], kernel_size, dilation=4, dropout=dropout),
            ConvBlock(channels[2], channels[1], kernel_size, dilation=2, dropout=dropout),
            ConvBlock(channels[1], channels[1], kernel_size, dilation=1, dropout=dropout),
            ConvBlock(channels[1], channels[0], kernel_size, dilation=1, dropout=dropout),
        )

        self.mask_head = nn.Sequential(
            nn.Conv1d(channels[0], in_bins, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, noisy_log_mag: torch.Tensor) -> torch.Tensor:
        """
        noisy_log_mag : [B, F, T]  — log1p(noisy_magnitude)
        Returns mask  : [B, F, T]  — values in (0, 1)
        """
        x = self.input_proj(noisy_log_mag)   # [B, C0, T]
        x = self.encoder(x)                  # [B, C2, T]
        x = self.bottleneck(x)               # [B, C2, T]
        x = self.decoder(x)                  # [B, C0, T]
        mask = self.mask_head(x)             # [B, F, T]
        mask= 0.05 + 0.95*mask  # floor at 0.1 to avoid zeroing out frequencies completely (helps stability)
        return mask

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enhance(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: takes LINEAR magnitude, returns enhanced LINEAR magnitude.
        noisy_mag: [B, F, T] or [F, T]
        """
        batched = noisy_mag.dim() == 3
        if not batched:
            noisy_mag = noisy_mag.unsqueeze(0)

        log_mag = torch.log1p(noisy_mag)
        mask = self.forward(log_mag)
        enhanced = mask * noisy_mag

        if not batched:
            enhanced = enhanced.squeeze(0)
        return enhanced


# ──────────────────────────────────────────────
#  Model factory
# ──────────────────────────────────────────────

def build_model(config: dict) -> SpeechEnhancementNet:
    m_cfg = config['model']
    model = SpeechEnhancementNet(
        in_bins=m_cfg['in_bins'],
        channels=m_cfg['channels'],
        kernel_size=m_cfg['kernel_size'],
        dropout=m_cfg['dropout'],
    )
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}  ({n_params/1e6:.2f}M)")
    return model


if __name__ == '__main__':
    # Quick sanity check
    model = SpeechEnhancementNet(in_bins=257)
    x = torch.randn(4, 257, 32)      # batch=4, 257 freq bins, 32 time frames
    mask = model(x)
    print(f"Input : {x.shape}")
    print(f"Mask  : {mask.shape}")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    print(f"Parameters: {model.count_parameters():,}")
