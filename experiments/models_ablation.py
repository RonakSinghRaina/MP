"""
Component-switchable variants of HybridRFINet, plus trivial references.

WHY THIS EXISTS
---------------
`hybrid_model.py`'s own docstring says of the deliberately-excluded extras:
"they belong in the ablation table as separate rows, not in v1". There is no
ablation table anywhere in this project, so nothing establishes that the three
components that ARE included do any work.

Worse, the hybrid differs from the tf_unet baseline in at least six ways at
once -- architecture, framework, loss (CE+Dice vs CE), normalisation
(GroupNorm vs none), padding (same vs valid), and output activation (raw logits
vs ReLU) -- plus batch size and features_root. Attributing the reported gap to
"the architecture" is not supported by any experiment that currently exists.

`ConfigurableUNet` keeps the exact skeleton of `HybridRFINet` and makes each
claimed component a switch, so `run_ablation.py` can turn one thing off at a
time while holding data, loss, optimiser and step budget fixed.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "src", "hybrid_rfi_package")))
from hybrid_model import HybridRFINet, ResBlock, MultiScaleStrip, ECA, count_parameters  # noqa: E402


class PlainBlock(nn.Module):
    """conv-norm-relu x2, with NO residual shortcut. Everything else identical.

    use_norm=False drops the GroupNorm layers, leaving conv-relu x2 -- the same
    block structure tf_unet uses, which has no normalisation anywhere. A bias is
    added back on the convolutions in that case, since without a following norm
    layer a bias-free conv cannot shift its output.
    """

    def __init__(self, in_ch, out_ch, dropout=0.0, use_norm=True):
        super().__init__()
        b = not use_norm

        def norm(c):
            return nn.GroupNorm(min(8, c), c) if use_norm else nn.Identity()

        self.b = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=b),
            norm(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=b),
            norm(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.b(x)


class PlainBlockNoNorm(PlainBlock):
    """PlainBlock with GroupNorm removed, as a two-argument Block factory."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__(in_ch, out_ch, dropout=dropout, use_norm=False)


class ConfigurableUNet(nn.Module):
    """
    HybridRFINet with each claimed component switchable.

    use_res   : residual shortcuts in the conv blocks   (claim 1 in hybrid_model.py)
    use_strip : multiscale anisotropic strip convolutions (claim 2)
    use_eca   : efficient channel attention             (claim 3)
    use_norm  : GroupNorm layers                        (NOT a claim -- a
                confound. tf_unet has no normalisation at all, so without this
                switch `plain_unet` still differs from the baseline by
                normalisation and the gap stays unattributed. PART 13.8 measured
                GroupNorm as doing heavy lifting: per-image normalisation costs
                tf_unet -0.244 but the hybrid only -0.077.)

    All three claims off == a plain GroupNorm U-Net of the same depth and width,
    trained with the same loss. That is the control the paper needs: it isolates
    the architecture from the framework/loss/normalisation changes that were
    made at the same time. Turning use_norm off as well moves one step closer
    still to tf_unet, leaving only padding, output activation and framework.
    """

    def __init__(self, in_channels=1, n_classes=2, base=32, depth=4, dropout=0.2,
                 use_res=True, use_strip=True, use_eca=True, use_norm=True):
        super().__init__()
        self.depth = depth
        if use_res and not use_norm:
            raise ValueError(
                "use_res=True with use_norm=False is not supported: ResBlock is "
                "imported unmodified from hybrid_model.py and has GroupNorm baked "
                "in. Removing normalisation is only defined for the plain block."
            )
        if use_strip and not use_norm:
            raise ValueError(
                "use_strip=True with use_norm=False is not supported: "
                "MultiScaleStrip is imported unmodified from hybrid_model.py and "
                "carries its own GroupNorm, so normalisation would not actually "
                "be removed."
            )
        Block = ResBlock if use_res else (PlainBlock if use_norm else PlainBlockNoNorm)
        self.enc, self.enc_ms, self.enc_eca = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ch, feats = in_channels, []
        for d in range(depth):
            out = base * (2 ** d)
            self.enc.append(Block(ch, out, dropout=dropout))
            self.enc_ms.append(MultiScaleStrip(out) if (use_strip and d < depth - 1) else nn.Identity())
            self.enc_eca.append(ECA(out) if use_eca else nn.Identity())
            feats.append(out)
            ch = out
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = Block(ch, ch * 2, dropout=dropout)
        self.bottleneck_ms = MultiScaleStrip(ch * 2) if use_strip else nn.Identity()
        bch = ch * 2
        self.up, self.dec, self.dec_eca = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for d in reversed(range(depth)):
            sc = feats[d]
            self.up.append(nn.ConvTranspose2d(bch, sc, 2, stride=2))
            self.dec.append(Block(sc * 2, sc, dropout=dropout))
            self.dec_eca.append(ECA(sc) if use_eca else nn.Identity())
            bch = sc
        self.head = nn.Conv2d(base, n_classes, 1)   # raw logits, as in HybridRFINet

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
            x = self.dec[i](torch.cat([s, x], dim=1))
            x = self.dec_eca[i](x)
        return self.head(x)


class TinyCNN(nn.Module):
    """
    A deliberately trivial reference: 2 conv layers + 1x1 head, ~2.6k parameters,
    no downsampling, no skip connections, no attention. If this lands anywhere
    near the reported baseline, the baseline is broken.
    """

    def __init__(self, base=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.GroupNorm(4, base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(4, base), nn.ReLU(inplace=True),
            nn.Conv2d(base, 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class LogisticPixel(nn.Module):
    """The absolute floor: a 1x1 conv, i.e. per-pixel logistic regression. 4 parameters."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(1, 2, 1)

    def forward(self, x):
        return self.c(x)


VARIANTS = {
    "hybrid_full":  dict(use_res=True,  use_strip=True,  use_eca=True,  use_norm=True),
    "no_strip":     dict(use_res=True,  use_strip=False, use_eca=True,  use_norm=True),
    "no_eca":       dict(use_res=True,  use_strip=True,  use_eca=False, use_norm=True),
    "no_res":       dict(use_res=False, use_strip=True,  use_eca=True,  use_norm=True),
    "plain_unet":   dict(use_res=False, use_strip=False, use_eca=False, use_norm=True),
    # One step closer to tf_unet than plain_unet: no normalisation anywhere.
    # What still differs from tf_unet after this is padding (same vs valid),
    # output activation (raw logits vs ReLU) and framework.
    "no_groupnorm": dict(use_res=False, use_strip=False, use_eca=False, use_norm=False),
}

ALL_NAMES = list(VARIANTS) + ["tiny_cnn", "logistic_pixel"]


def build(name, base=32, depth=4, dropout=0.2):
    if name == "tiny_cnn":
        return TinyCNN()
    if name == "logistic_pixel":
        return LogisticPixel()
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {ALL_NAMES}")
    return ConfigurableUNet(base=base, depth=depth, dropout=dropout, **VARIANTS[name])


if __name__ == "__main__":
    ref = HybridRFINet(1, 2, base=32, depth=4, dropout=0.2)
    print(f"{'variant':<18}{'params':>12}")
    print(f"{'HybridRFINet':<18}{count_parameters(ref):>12,}   <- the published model")
    for n in ALL_NAMES:
        print(f"{n:<18}{count_parameters(build(n)):>12,}")
    print("\nSanity: hybrid_full must equal HybridRFINet ->",
          count_parameters(build('hybrid_full')) == count_parameters(ref))
