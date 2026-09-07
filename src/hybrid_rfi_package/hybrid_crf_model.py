"""
HybridCRFNet — HybridRFINet with a differentiable CRF refinement head.

Motivated by Chen et al. 2023, RAA 23 104004, "Cleaning Radio Frequency
Interference in Pulsar-Folded Data Based on the Conditional Random Fields with
an Adaptive Prior" (the "2sigmaCRF" method), doi:10.1088/1674-4527/acd52b.
Paper copy: notes/Chen2023_2sigmaCRF_pulsar_RFI.pdf

WHY THIS EXISTS
---------------
PART 15 measured exactly where our F1 is lost. The model already catches 93.7%
of bright RFI, but only 25.4% of RFI that is DIMMER THAN A TYPICAL CLEAN PIXEL
— and that tier is 20% of all RFI, worth +0.098 F1 if solved. Those pixels
carry no per-pixel brightness evidence by construction; the only signal is
their neighbours. Chen et al.'s abstract describes precisely this: their method
recovers "weak RFIs that are unrecognizable in some pixels but picked out based
on their neighborhoods."

WHAT CHEN ET AL. ACTUALLY DO (paper sections 3.1-3.2)
-----------------------------------------------------
Two stages, no neural network anywhere:

1. **Adaptive "2sigma" prior.** Histogram the per-pixel rms values, fit a
   Gaussian, call everything within +-2sigma of the peak "normal" and beyond
   3sigma "RFI". The threshold adapts to the data rather than being fixed.

2. **Dense CRF refinement.** Minimise the Gibbs energy

       U(w) = sum_i V1(w_i) + sum_(i,i') V2(w_i, w_i')                   (their Eq. 2)

   with the unary V1 = -log P(pixel i is class w_i) from the Gaussian fit, and
   the pairwise term

       V2(w_i,w_i') = mu(w_i,w_i') * ( lambda_a * k_a + lambda_s * k_s )  (Eq. 3)
       k_a = exp( -|p_i-p_j|^2 / 2*xi1^2  -  |I_i-I_j|^2 / 2*xi2^2 )      (Eq. 4)
       k_s = exp( -|p_i-p_j|^2 / 2*xi3^2 )                                (Eq. 5)

   mu(w_i,w_i') = 1[w_i != w_i'] is the Potts penalty: pay only when two
   pixels that look alike and sit close together are given DIFFERENT labels.
   k_a is the "appearance" kernel (position AND intensity), k_s the
   "smoothness" kernel (position only). They use fully-connected CRFs
   (Krahenbuhl & Koltun 2011, denseCRF/SimpleCRF), 5 iterations, with
   lambda_a=3, lambda_s=20, xi1=1, xi2=10, xi3=1.

WHAT WE DO DIFFERENTLY, AND WHY
--------------------------------
1. **The unary comes from the CNN, not from a Gaussian fit.** Our backbone
   already produces far better per-pixel evidence than a histogram threshold
   (ROC 0.94). Chen et al. had no learned model; we do. The CRF's job here is
   only to fix the neighbourhood-inconsistent pixels.

2. **Mean-field inference unrolled as differentiable layers**, so the CRF
   trains end-to-end with the backbone instead of being a fixed post-process.
   This is the CRF-as-RNN idea (Zheng et al. 2015) applied to Chen's energy.

3. **Local window instead of fully-connected.** A truly dense CRF needs
   permutohedral-lattice filtering. Restricting the pairwise term to a k x k
   window makes message passing a plain convolution (the ConvCRF
   simplification, Teichmann & Cipolla 2018) — far cheaper, pure PyTorch, and
   adequate because the neighbourhood evidence we need is local along each
   axis.

4. **ANISOTROPIC spatial kernels — our adaptation, not in the paper.** Chen
   et al. use one isotropic xi for position. But PART 11.2 measured that LOFAR
   RFI is strongly column-concentrated: 65-74% of flagged pixels sit in 16 of
   512 columns, because a transmitter occupies a fixed frequency and persists
   in time. So this model learns SEPARATE bandwidths for the time axis (rows)
   and the frequency axis (columns), letting the neighbourhood stretch along
   whichever axis the RFI actually extends. This is the same physical argument
   that motivates the strip convolutions in the backbone.

5. **xi2 rescaled.** Chen et al. use xi2=10 on raw rms values. Our inputs are
   normalised to [0,1], so an intensity bandwidth of 10 would make the
   appearance kernel constant. Initialised to 0.1 here and learned.

6. **The 2sigma adaptive prior is kept as an optional extra unary term**
   (`use_sigma_prior`). It is computed per image from the intensity
   distribution, costs no parameters beyond a single learnable weight, and is
   the part of Chen et al. that is genuinely theirs rather than generic CRF.

MEASURED BEFORE ANY TRAINING (frozen trained base-8 backbone, 109 test images)
------------------------------------------------------------------------------
| configuration                          | max F1 |
|----------------------------------------|--------|
| backbone alone                         | 0.6592 |
| + CRF at Chen et al.'s lambda (3, 20)  | 0.4884 |
| + CRF, lambda (3, 5)                   | 0.6162 |
| + CRF, lambda (1, 0.1)  [the default]  | 0.6594 |
| + CRF, lambda (1, 0.1), anisotropic    | 0.6604 |

So as a **post-hoc refinement on a frozen backbone the CRF buys essentially
nothing** -- at best +0.0012, well inside the 0.0040 seed spread. Two reasons,
both worth stating in the paper rather than hiding:

1. Our backbone already reasons about neighbourhoods, through its strip
   convolutions and U-Net receptive field. Chen et al. applied their CRF to a
   bare histogram threshold with no spatial modelling at all, so it had far
   more to fix.
2. A CRF can only *redistribute* evidence that is already in the unary. The
   PART 15 problem is that faint RFI has almost no evidence in the unary to
   redistribute.

The untested case, and the reason this model exists, is **end-to-end
training**: with the CRF differentiable, the backbone can learn to emit
unaries shaped to be refined, rather than unaries that are already final.
That is the CRF-as-RNN argument (Zheng et al. 2015) and it cannot be
evaluated by bolting the CRF on afterwards. Expectations should be modest.

PARAMETER COST
--------------
The CRF head adds a handful of scalars — kernel weights, bandwidths, the
compatibility matrix and the prior weight. Call `HybridCRFNet(...).crf_parameter_count()`
for the exact number; it is under 20, i.e. ~0.003% of the base-8 backbone's
593,842. This was a hard requirement: the project's own PART 6/13.7 finding is
that capacity is NOT the bottleneck, so a refinement head that added real
parameters would undercut the paper's own argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_model import HybridRFINet


class MeanFieldCRF(nn.Module):
    """Differentiable mean-field inference for Chen et al.'s energy, in a k x k window.

    One mean-field iteration is

        Q_i(l)  <-  softmax( -U_i(l) - sum_{l'} mu(l,l') * sum_{j in N(i)} K(i,j) Q_j(l') )

    where K(i,j) = lambda_a * k_a(i,j) + lambda_s * k_s(i,j) exactly as in the
    paper's Eq. 3-5, and mu is the learned label-compatibility (Potts at init).

    All kernel weights and bandwidths are stored as logs so they stay positive
    under gradient descent.
    """

    def __init__(self, n_classes=2, kernel_size=7, n_iters=5,
                 lambda_a=1.0, lambda_s=0.1,
                 xi_time=0.5, xi_freq=5.0, xi_intensity=0.1, xi_smooth=1.0,
                 learn_kernels=True):
        """NOTE ON DEFAULTS -- these are NOT Chen et al.'s values, deliberately.

        Chen et al. use lambda_a=3, lambda_s=20 on FAST pulsar-folded data.
        Applied unchanged to our LOFAR model those values are **destructive**:
        measured on the frozen trained base-8 backbone, max F1 falls from
        0.6592 to 0.4884, because precision rises to 0.742 while recall
        collapses to 0.347. The CRF erases isolated detections whose
        neighbours are labelled clean. Their RFI arrives in large contiguous
        blocks; ours is sparse and thin at 0.77% prevalence, so strong
        smoothing is exactly wrong. The paper anticipates this -- "new
        parameters may need to be tested and selected for data obtained from
        other telescopes".

        The defaults here are the best point found in a sweep on the frozen
        backbone. Pass chen_init=True to the wrapper for the paper's values.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.n_classes = n_classes
        self.k = kernel_size
        self.pad = kernel_size // 2
        self.n_iters = n_iters

        def p(v):
            t = torch.tensor(float(v)).log()
            return nn.Parameter(t, requires_grad=learn_kernels)

        # Eq. 3 kernel weights
        self.log_lambda_a = p(lambda_a)
        self.log_lambda_s = p(lambda_s)
        # Eq. 4/5 bandwidths. Separate rows (time) and cols (frequency): ours.
        self.log_xi_time = p(xi_time)
        self.log_xi_freq = p(xi_freq)
        self.log_xi_intensity = p(xi_intensity)
        self.log_xi_smooth = p(xi_smooth)

        # label compatibility mu(l,l'), initialised to Potts: 0 on the diagonal,
        # 1 off it, so only disagreement between neighbours costs energy.
        potts = 1.0 - torch.eye(n_classes)
        self.compat = nn.Parameter(potts.clone(), requires_grad=learn_kernels)

        # relative offsets of the window, registered so .to(device) moves them
        ar = torch.arange(kernel_size) - self.pad
        dy = ar.view(-1, 1).expand(kernel_size, kernel_size).reshape(-1).float()
        dx = ar.view(1, -1).expand(kernel_size, kernel_size).reshape(-1).float()
        self.register_buffer("dy", dy)          # row offset  = TIME
        self.register_buffer("dx", dx)          # col offset  = FREQUENCY
        centre = (kernel_size * kernel_size) // 2
        m = torch.ones(kernel_size * kernel_size)
        m[centre] = 0.0                          # j != i in the paper's sum
        self.register_buffer("self_mask", m)

    def _pairwise_kernel(self, img):
        """K(i,j) for every offset j in the window. Returns (B, k*k, H, W)."""
        B, _, H, W = img.shape
        xi_t = self.log_xi_time.exp().clamp(min=1e-3)
        xi_f = self.log_xi_freq.exp().clamp(min=1e-3)
        xi_i = self.log_xi_intensity.exp().clamp(min=1e-3)
        xi_s = self.log_xi_smooth.exp().clamp(min=1e-3)

        # anisotropic squared distance, ours; Chen et al. use one isotropic xi
        d2_aniso = (self.dy / xi_t) ** 2 + (self.dx / xi_f) ** 2       # (k*k,)
        d2_smooth = (self.dy ** 2 + self.dx ** 2) / (xi_s ** 2)        # (k*k,)

        # |I_i - I_j|^2 for every offset
        nb = F.unfold(img, self.k, padding=self.pad)                   # (B,k*k,H*W)
        nb = nb.view(B, self.k * self.k, H, W)
        di2 = (nb - img) ** 2                                          # broadcast centre

        shape = (1, self.k * self.k, 1, 1)
        k_a = torch.exp(-0.5 * d2_aniso.view(shape) - di2 / (2 * xi_i ** 2))
        k_s = torch.exp(-0.5 * d2_smooth.view(shape)).expand_as(k_a)

        K = self.log_lambda_a.exp() * k_a + self.log_lambda_s.exp() * k_s
        return K * self.self_mask.view(shape)

    def forward(self, unary_logits, img):
        """unary_logits (B,C,H,W) from the backbone; img (B,1,H,W) normalised input.

        Optimisation for the binary case: Q sums to 1 over classes, so with
        C=2 only ONE channel needs the expensive spatially-varying message
        pass. The other follows from
            msg_0 = sum_j K(i,j) * (1 - Q_j(1)) = Ksum_i - msg_1
        with Ksum precomputed once. Halves the work and the peak memory, and
        is exact -- not an approximation.
        """
        B, C, H, W = unary_logits.shape
        K = self._pairwise_kernel(img)                                 # (B,k*k,H,W)
        Q = F.softmax(unary_logits, dim=1)

        if C == 2:
            Ksum = K.sum(dim=1, keepdim=True)                          # (B,1,H,W)
            for _ in range(self.n_iters):
                q1 = Q[:, 1:2]
                qu = F.unfold(q1, self.k, padding=self.pad).view(B, self.k * self.k, H, W)
                msg1 = (qu * K).sum(dim=1, keepdim=True)               # (B,1,H,W)
                msg = torch.cat([Ksum - msg1, msg1], dim=1)            # (B,2,H,W)
                compat = torch.einsum("lm,bmhw->blhw", self.compat, msg)
                Q = F.softmax(unary_logits - compat, dim=1)
        else:
            for _ in range(self.n_iters):
                Qu = F.unfold(Q, self.k, padding=self.pad)
                Qu = Qu.view(B, C, self.k * self.k, H, W)
                msg = (Qu * K.unsqueeze(1)).sum(dim=2)
                compat = torch.einsum("lm,bmhw->blhw", self.compat, msg)
                Q = F.softmax(unary_logits - compat, dim=1)

        return torch.log(Q.clamp_min(1e-8))        # logits again, for CE loss


class SigmaPrior(nn.Module):
    """Chen et al.'s adaptive '2sigma' prior (their section 3.1), as a unary bias.

    Per image: take the median and a robust sigma (MAD) of the intensities,
    then score each pixel by how many sigma it sits from the centre. Pixels
    beyond `n_sigma` get pushed toward the RFI class. Parameter-free apart from
    one learnable weight, and adaptive to each image exactly as the paper
    intends.

    NOTE this is a WEAK prior on LOFAR: PART 11.7 measured that ~24% of true
    RFI is dimmer than the median clean pixel, so a magnitude threshold cannot
    see it. It is included because it is the distinctive part of Chen et al.,
    and because the CRF can override it — but it is not expected to carry the
    result on its own.
    """

    def __init__(self, n_sigma=2.0, weight=1.0):
        super().__init__()
        self.n_sigma = n_sigma
        self.log_weight = nn.Parameter(torch.tensor(float(weight)).log())

    def forward(self, img):
        B = img.shape[0]
        flat = img.view(B, -1)
        med = flat.median(dim=1, keepdim=True).values
        mad = (flat - med).abs().median(dim=1, keepdim=True).values * 1.4826 + 1e-6
        z = ((flat - med).abs() / mad).view_as(img)                    # (B,1,H,W)
        excess = F.relu(z - self.n_sigma)                              # 0 inside +-2 sigma
        w = self.log_weight.exp()
        bias = torch.cat([torch.zeros_like(excess), w * excess], dim=1)
        return bias                                                    # (B,2,H,W)


class HybridCRFNet(nn.Module):
    """HybridRFINet + Chen-style CRF refinement. The backbone is untouched."""

    def __init__(self, in_channels=1, n_classes=2, base=8, depth=4, dropout=0.2,
                 crf_kernel=7, crf_iters=5, use_sigma_prior=True,
                 n_sigma=2.0, learn_kernels=True, chen_init=False):
        super().__init__()
        self.backbone = HybridRFINet(in_channels, n_classes, base=base,
                                     depth=depth, dropout=dropout)
        kw = dict(lambda_a=3.0, lambda_s=20.0, xi_time=1.0, xi_freq=1.0,
                  xi_intensity=0.1, xi_smooth=1.0) if chen_init else {}
        self.crf = MeanFieldCRF(n_classes=n_classes, kernel_size=crf_kernel,
                                n_iters=crf_iters, learn_kernels=learn_kernels, **kw)
        self.sigma_prior = SigmaPrior(n_sigma=n_sigma) if use_sigma_prior else None

    def forward(self, x, return_unary=False):
        unary = self.backbone(x)
        if self.sigma_prior is not None:
            unary = unary + self.sigma_prior(x)
        out = self.crf(unary, x)
        return (out, unary) if return_unary else out

    # ---- introspection ----------------------------------------------------
    def crf_parameter_count(self):
        n = sum(p.numel() for p in self.crf.parameters())
        if self.sigma_prior is not None:
            n += sum(p.numel() for p in self.sigma_prior.parameters())
        return n

    def backbone_parameter_count(self):
        return sum(p.numel() for p in self.backbone.parameters())

    def load_backbone(self, state_dict, strict=True):
        """Warm-start from an existing trained HybridRFINet checkpoint."""
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def describe(self):
        b, c = self.backbone_parameter_count(), self.crf_parameter_count()
        return (f"HybridCRFNet: backbone {b:,} + CRF head {c} = {b + c:,} parameters "
                f"({100.0 * c / (b + c):.4f}% in the CRF)")
