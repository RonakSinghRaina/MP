"""
Hybrid U-Net for RFI segmentation -- architecture definition only.

DESIGN RATIONALE (every component justified, none decorative)
--------------------------------------------------------------
Measured on this dataset: weak RFI sits at ~1.5 sigma contrast above the
background after normalization, strong RFI at ~9.7 sigma. At 1.5 sigma a
single pixel is statistically indistinguishable from a noise fluctuation.
Therefore weak RFI can ONLY be detected from SPATIAL COHERENCE -- a faint
narrowband line is detectable because it spans hundreds of time bins, not
because any one of its pixels is bright. Every component below exists to
serve that finding.

1. RESIDUAL BLOCKS  (evidence: RFI-Net, Yang et al. 2020, MNRAS 492, 1421 --
   reported better RFI identification with finer edges and lower false
   positives than plain U-Net)
   -> Identity shortcuts let low-contrast features survive depth instead of
      being washed out by repeated convolution.

2. MULTISCALE *STRIP* CONVOLUTIONS  (evidence: EMSCA-UNet, Gu et al. 2024,
   MNRAS 529, 4719 -- multiscale convolutional attention, reported to beat
   U-Net / RFI-Net / R-Net)
   -> Standard square kernels are a poor match for RFI morphology. In this
      dataset, narrowband RFI = thin HORIZONTAL lines spanning the full time
      axis; wideband bursts = thin VERTICAL columns spanning frequency.
      Anisotropic strip kernels (1xK and Kx1) integrate signal ALONG those
      structures, which is precisely how a 1.5-sigma line becomes detectable:
      averaging along K pixels of a coherent line grows SNR ~sqrt(K) while
      noise averages down. Several K values run in parallel because RFI
      structures occur at different widths/durations.

3. EFFICIENT CHANNEL ATTENTION (ECA)  (evidence: EMSCA-UNet, above)
   -> After multiscale extraction, some channels carry the low-contrast
      evidence and some carry bright-RFI evidence. ECA reweights channels
      with negligible parameter cost (a single 1D conv), so faint-structure
      channels are not drowned out by high-amplitude ones.

4. GROUPNORM, *NOT* BATCHNORM  (evidence: this project's own earlier failure)
   -> An earlier PyTorch attempt on this exact task used BatchNorm at
      batch_size=1. BatchNorm estimates mean/variance ACROSS the batch; with
      one sample those statistics are noise, which destabilises training.
      Since a 6GB GPU forces batch_size=1 at 512x512, GroupNorm (which
      normalises over channel groups within a single sample) is the correct
      choice and is batch-size independent.

5. NO ReLU ON THE OUTPUT LOGITS  (evidence: this project's own earlier failure)
   -> tf_unet applies ReLU to its final logits. When both logits go negative,
      ReLU clamps them to zero, softmax gives exactly [0.5, 0.5], and the
      gradient is exactly zero -- an irrecoverable dead network. This was
      measured killing training at iteration 4 with the paper's lr=0.2. This
      model emits raw logits, so that failure mode cannot occur.

6. SAME-PADDING + a modest depth of 4
   -> Keeps output size equal to input size (simpler, no crop bookkeeping)
      and keeps activation memory within a 6GB budget at 512x512.

DELIBERATELY *NOT* INCLUDED
-----------------------------
- Transformers / self-attention: quadratic memory cost is incompatible with
  512x512 patches on 6GB, and there is no RFI-specific evidence they beat
  the cheaper mechanisms above.
- Deep supervision, boundary loss, restoration branch: each is defensible,
  but adding them all at once makes it impossible to attribute any gain.
  They belong in the ablation table as separate rows, not in v1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient Channel Attention -- channel reweighting via one 1D conv."""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                                  # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))        # B,1,C
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))   # B,C,1,1
        return x * y.expand_as(x)


class MultiScaleStrip(nn.Module):
    """
    Parallel anisotropic strip convolutions matched to RFI morphology.

    Horizontal strips (1xK) integrate along the TIME axis -> narrowband and
    persistent-band RFI. Vertical strips (Kx1) integrate along the FREQUENCY
    axis -> wideband bursts and broadband blocks. Multiple K values cover
    different durations/bandwidths. Depthwise so the cost stays small.
    """

    def __init__(self, channels, kernel_sizes=(7, 11, 21)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, channels, (1, k), padding=(0, k // 2), groups=channels, bias=False),
                nn.Conv2d(channels, channels, (k, 1), padding=(k // 2, 0), groups=channels, bias=False),
            ))
        self.fuse = nn.Conv2d(channels * (len(kernel_sizes) + 1), channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        outs = [x] + [b(x) for b in self.branches]
        return self.act(self.norm(self.fuse(torch.cat(outs, dim=1))))


class ResBlock(nn.Module):
    """Residual conv block with GroupNorm (batch-size independent)."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, 1, bias=False))

    def forward(self, x):
        idt = self.skip(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return self.act(y + idt)


class HybridRFINet(nn.Module):
    """
    Encoder-decoder with residual blocks, multiscale strip convolutions and
    efficient channel attention. Emits RAW LOGITS (no output ReLU).
    """

    def __init__(self, in_channels=1, n_classes=2, base=32, depth=4, dropout=0.2):
        super().__init__()
        self.depth = depth

        self.enc = nn.ModuleList()
        self.enc_ms = nn.ModuleList()
        self.enc_eca = nn.ModuleList()
        ch = in_channels
        feats = []
        for d in range(depth):
            out = base * (2 ** d)
            self.enc.append(ResBlock(ch, out, dropout=dropout))
            # Strip convs are most valuable at high resolution, where thin
            # low-contrast structures still exist; skip them at the deepest
            # encoder level where features are already coarse.
            self.enc_ms.append(MultiScaleStrip(out) if d < depth - 1 else nn.Identity())
            self.enc_eca.append(ECA(out))
            feats.append(out)
            ch = out

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ResBlock(ch, ch * 2, dropout=dropout)
        self.bottleneck_ms = MultiScaleStrip(ch * 2)
        bch = ch * 2

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.dec_eca = nn.ModuleList()
        for d in reversed(range(depth)):
            skip_ch = feats[d]
            self.up.append(nn.ConvTranspose2d(bch, skip_ch, kernel_size=2, stride=2))
            self.dec.append(ResBlock(skip_ch * 2, skip_ch, dropout=dropout))
            self.dec_eca.append(ECA(skip_ch))
            bch = skip_ch

        # Raw logits -- deliberately NO activation here (see module docstring)
        self.head = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for d in range(self.depth):
            x = self.enc[d](x)
            x = self.enc_ms[d](x)
            x = self.enc_eca[d](x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck_ms(self.bottleneck(x))

        for i, d in enumerate(reversed(range(self.depth))):
            x = self.up[i](x)
            s = skips[d]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([s, x], dim=1)
            x = self.dec[i](x)
            x = self.dec_eca[i](x)

        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = HybridRFINet()
    print(f"HybridRFINet parameters: {count_parameters(m):,}")
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = m(x)
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}")
