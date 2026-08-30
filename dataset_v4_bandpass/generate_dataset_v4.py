"""
Synthetic RFI Spectrogram Generator — v4 (instrument bandpass)
==============================================================

v4 = v3 (`hybrid_rfi_package/dataset_generator_v3_strength.py`) plus a physically
explicit receiver bandpass, drawn fresh for every image, applied multiplicatively
to everything that reaches the receiver.

WHAT CHANGED FROM v3, AND WHY
-----------------------------

1. A real multiplicative bandpass B(f).
   v3's `generate_bandpass_gain_curve` was not a bandpass. It set the *additive*
   background pedestal (`pure = gain[:,None] + sigma[:,None]*noise`) and left RFI
   untouched. v4 keeps that curve as the sky/system pedestal but adds a genuine
   multiplicative gain applied to sky, noise and RFI alike — one signal chain,
   nothing exempt.

2. A post-gain receiver noise floor. THIS IS NOT OPTIONAL COSMETICS.
   If B(f) multiplies the RFI and its noise equally, it cancels out of their
   ratio:  (B*A)/(B*sigma) = A/sigma.  Detectability would be *exactly* preserved
   at every channel for every value of B, including B=0.001, and the whole
   band-edge labelling problem would silently be a no-op. That is also why
   bandpass calibration works in real astronomy: dividing out B recovers uniform
   sensitivity.
   Real instruments still discard edge channels because amplifier and digitiser
   noise enter *after* the gain stage and do not scale with it. Where B is small
   the amplified sky sinks toward that fixed floor:

       SNR(f) = snr_injected * B(f)*sigma_sky / sqrt(B(f)^2*sigma_sky^2 + sigma_rx^2)

   which is flat in mid-band and falls to zero at the edges. `--rx_noise_frac 0`
   recovers the pure-radiometer model where the bandpass changes brightness but
   never detectability.

3. The label is now a threshold, not a union.
   v3: `mask = hard_geometric_footprint OR (blurred_rfi > 0.5*sigma)`. Because it
   was a union it could only ever ADD pixels, so a pixel could be labelled RFI
   while carrying almost no RFI power. (Measured on the v3 train split: 0.06% of
   labelled pixels sit below 0.5 sigma, worst image 4.49%.)
   v4: `mask = (B*blurred_rfi) > threshold * sigma_total`, evaluated AFTER the
   bandpass and against the LOCAL total noise of that channel. Band-edge RFI that
   the instrument genuinely buried is therefore not labelled — labels stay
   satisfiable, and reported scores keep meaning something.

4. `strength/` is now post-bandpass.
   v3 stored `rfi_layer / sigma_local` (pre-bandpass). v4 stores
   `(B*rfi_layer) / sigma_total`, so `strength > --mask_sigma_threshold` is
   exactly the mask condition and strength-stratified evaluation still answers
   "how visible was this RFI".

5. `<split>/bandpass/bandpass_NNNN.npy` — the exact B(f) applied to each image.

6. v3's `generate_edge_rolloff` is OFF by default (`--v3_edge_rolloff`). The new
   Tukey taper does that job explicitly; running both would attenuate the edges
   twice and make `--edge_frac` mean something other than what it says.

NOT changed: the array convention (axis 0 = frequency, axis 1 = time), the six
RFI morphologies, the density tiers, the `_scale_range` fractional-size scaling,
the file layout, and the 11 metadata fields the training scripts read.

ZERO HARDCODED NUMBERS
----------------------
Every constant is an argparse flag with a documented default. Defaults reproduce
v3's behaviour for everything that carries over, so any difference between v3 and
v4 output is attributable to the bandpass, not to drift in the tuning.

REPRODUCIBILITY
---------------
Seeded once via `np.random.seed`, then all draws come from numpy's legacy
RandomState in a fixed call order. RandomState is covered by numpy's stream
compatibility guarantee, so this reproduces bit-exactly across numpy versions —
which the modern `default_rng` does not promise. Verified by generating the whole
dataset twice and comparing SHA-256 of every file (see README).

Usage:
    python3 generate_dataset_v4.py --output_dir "../Synthetic Dataset 1024x265" --seed 42
    python3 generate_dataset_v4.py --output_dir /tmp/sample --n_images 10 --n_previews 10
"""

import os
import json
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")          # no display on a headless/CLI run
import matplotlib.pyplot as plt
from copy import copy


# Pixel-valued morphology parameters are quoted at this reference dimension and
# rescaled to the actual image size, so RFI structures stay at constant
# FRACTIONAL size and datasets at different resolutions remain comparable.
# Parameters expressed in MHz are already resolution-independent and are not
# scaled.
REFERENCE_DIM = 1024

DENSITY_TIERS = ("clean", "light", "moderate", "heavy")


def _scale_range(lo, hi, n, reference=REFERENCE_DIM, min_lo=1):
    """Scale an integer pixel range [lo, hi) by n / reference."""
    f = n / float(reference)
    slo = max(min_lo, int(round(lo * f)))
    shi = max(slo + 1, int(round(hi * f)))
    return slo, shi


def _u(rng_range):
    """Uniform draw from a (lo, hi) pair. Collapses to a constant when lo == hi."""
    lo, hi = rng_range
    return lo if lo == hi else np.random.uniform(lo, hi)


def _log_uniform(rng_range):
    """Log-uniform draw — equal probability per decade, so weak RFI is not rare."""
    lo, hi = rng_range
    return float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))


def _randint(rng_range):
    """Inclusive-low, exclusive-high integer draw from a (lo, hi) pair."""
    lo, hi = rng_range
    return lo if hi <= lo + 1 else int(np.random.randint(lo, hi))


# ============================================================================
# Section 1: Sky + receiver noise (the v3 pipeline, parameterised)
# ============================================================================

def generate_sky_pedestal(n_freq, p):
    """
    Smooth spectral energy distribution of the sky + system temperature.

    This is v3's `generate_bandpass_gain_curve`, renamed. Despite the old name it
    is an ADDITIVE pedestal that sets the mean background level per channel, not
    a gain. The actual receiver gain in v4 is `generate_bandpass` below.
    """
    freqs = np.linspace(p.freq_lo_mhz, p.freq_hi_mhz, n_freq)

    center = _u(p.sed_center_mhz)
    width = _u(p.sed_width_mhz)
    amplitude = _u(p.sed_amp)

    pedestal = amplitude * np.exp(-0.5 * ((freqs - center) / width) ** 2)

    # Small-scale gain ripple of the analogue chain.
    n_ripples = _randint(p.sed_ripple_modes)
    ripple_amp = _u(p.sed_ripple_frac) * amplitude
    ripple_phase = np.random.uniform(0, 2 * np.pi)
    pedestal += ripple_amp * np.sin(
        n_ripples * np.pi * np.linspace(0, 1, n_freq) + ripple_phase)

    return np.maximum(pedestal, p.sed_floor)


def generate_v3_edge_rolloff(n_freq, p):
    """
    v3's linear band-edge attenuation. Retained only so `--v3_edge_rolloff` can
    reproduce v3 exactly; off by default because the Tukey taper in
    `generate_bandpass` supersedes it and stacking both double-attenuates.
    """
    lo, hi = _scale_range(*p.v3_rolloff_channels, n=n_freq)
    w = _randint((lo, hi))

    edge = np.ones(n_freq)
    edge[:w] = np.linspace(p.v3_rolloff_min, 1.0, w)
    edge[-w:] = np.linspace(1.0, p.v3_rolloff_min, w)
    return edge


def generate_temporal_drift(n_time, p):
    """Slow gain drift over an observation — atmosphere, elevation, electronics warming."""
    amp = _u(p.drift_amp)
    n_modes = _randint(p.drift_modes)

    drift = np.ones(n_time)
    for _ in range(n_modes):
        period = _u(p.drift_period_frac) * n_time
        phase = np.random.uniform(0, 2 * np.pi)
        drift += amp / n_modes * np.sin(2 * np.pi * np.arange(n_time) / period + phase)
    return drift


def _gaussian_kernel_1d(sigma, truncate=4.0):
    """
    The discrete kernel `scipy.ndimage.gaussian_filter` actually applies.

    Needed because the correlation smoothing in `generate_pure_signal` reduces
    the per-pixel noise sd below the nominal sigma, and the label threshold is
    measured against the REALISED noise. Reproduces scipy's own radius and
    normalisation so the correction is exact rather than approximate.
    """
    radius = int(truncate * sigma + 0.5)
    if radius < 1:
        return np.ones(1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def generate_pure_signal(n_freq, n_time, p):
    """
    Thermal-noise background before the receiver gain is applied.

    Returns:
        pure_signal   (n_freq, n_time) float64 — pedestal + correlated noise
        sky_sigma     (n_freq,)  nominal per-channel noise sd (RFI is quoted against this)
        sky_sigma_eff (n_freq,)  REALISED per-channel noise sd after drift + smoothing
    """
    sigma_bg = _u(p.sigma_bg)

    pedestal = generate_sky_pedestal(n_freq, p)
    if p.v3_edge_rolloff:
        pedestal = pedestal * generate_v3_edge_rolloff(n_freq, p)

    # Heteroscedastic noise. NOTE this is v3's convention and it is the opposite
    # of a radiometer (noise rises where the pedestal is LOW). Deliberately kept:
    # changing it would move every image statistic and break comparability with
    # the published v3 results. The new B(f) is a separate layer on top.
    pedestal_norm = pedestal / np.max(pedestal)
    sky_sigma = sigma_bg / np.sqrt(np.maximum(pedestal_norm, p.hetero_floor))

    noise = np.random.normal(0, 1, size=(n_freq, n_time))
    pure_signal = pedestal[:, np.newaxis] + sky_sigma[:, np.newaxis] * noise

    drift = generate_temporal_drift(n_time, p)
    pure_signal *= drift[np.newaxis, :]

    corr_f = _u(p.corr_sigma_freq)
    corr_t = _u(p.corr_sigma_time)
    pure_signal = gaussian_filter(pure_signal, sigma=(corr_f, corr_t))

    # Realised noise sd per channel. Smoothing white noise by a normalised kernel
    # k scales its variance by sum(k^2); smoothing across frequency also mixes
    # channels of differing sigma, which is exactly a convolution of the variance
    # profile. Drift multiplies signal and noise together, contributing
    # mean(drift^2) once averaged over time. All three terms are exact, so the
    # threshold is measured against the true local noise rather than a nominal.
    kf = _gaussian_kernel_1d(corr_f)
    kt = _gaussian_kernel_1d(corr_t)
    var_profile = np.convolve(sky_sigma ** 2, kf ** 2, mode="same")
    var_eff = var_profile * np.sum(kt ** 2) * np.mean(drift ** 2)
    sky_sigma_eff = np.sqrt(var_eff)

    return pure_signal, sky_sigma, sky_sigma_eff


# ============================================================================
# Section 2: THE NEW PHYSICS — receiver bandpass B(f)
# ============================================================================

def tukey_taper(n_freq, edge_frac):
    """
    Raised-cosine (Tukey) passband envelope: flat in the middle, smooth roll-off
    over `edge_frac` of the band at each edge.

    Ramps from just above zero rather than exactly zero, so no channel is
    identically dead and B(f) stays safe to divide by.
    """
    w = np.ones(n_freq)
    n_edge = int(round(edge_frac * n_freq))
    if n_edge < 1:
        return w
    n_edge = min(n_edge, n_freq // 2)

    x = np.arange(1, n_edge + 1) / (n_edge + 1.0)
    ramp = 0.5 * (1.0 - np.cos(np.pi * x))
    w[:n_edge] = ramp
    w[n_freq - n_edge:] = ramp[::-1]
    return w


def _jittered(value, jitter):
    """
    Scale `value` by a fresh multiplicative draw in [1-jitter, 1+jitter].

    This is what makes the bandpass vary BETWEEN images: same family, same
    nominal scale, different realisation each time. At jitter=0 the amplitudes
    are identical across images but the phases still differ, so the curves are
    never the single memorisable shape a fixed bandpass would give.
    """
    if jitter <= 0:
        return value
    return value * (1.0 + jitter * np.random.uniform(-1.0, 1.0))


def generate_bandpass(n_freq, p):
    """
    Multiplicative receiver gain B(f), one value per frequency channel, drawn
    fresh for this image. Three components, each independently flagged:

      1. PASSBAND ENVELOPE  — Tukey raised-cosine roll-off at both edges.
      2. SLOW RIPPLE        — sum of low-order sinusoids from imperfect filters.
      3. STANDING WAVE      — fixed-period sinusoid from reflections in the path.

    Returns B (n_freq,) float64, and a dict of the drawn parameters for metadata.
    """
    channels = np.arange(n_freq)

    # 1. Passband envelope.
    edge_frac = float(np.clip(_jittered(p.edge_frac, p.bandpass_jitter), 0.0, 0.5))
    envelope = tukey_taper(n_freq, edge_frac)

    # 2. Slow ripple. `--ripple_amp` is fractional PEAK-TO-PEAK: dividing by
    #    n_modes bounds the worst case (all modes in phase) at exactly that
    #    value; random phases make the typical excursion smaller.
    n_modes = _randint(p.ripple_modes)
    ripple_amp = _jittered(p.ripple_amp, p.bandpass_jitter)
    ripple = np.zeros(n_freq)
    for _ in range(n_modes):
        period = _u(p.ripple_period_frac) * n_freq
        phase = np.random.uniform(0, 2 * np.pi)
        ripple += np.sin(2 * np.pi * channels / period + phase) / n_modes
    ripple *= ripple_amp / 2.0

    # 3. Standing wave — a FIXED period in frequency, unlike the ripple's
    #    band-scale variation. Period jitters slightly between images because
    #    path length is not identical observation to observation.
    sw_period = max(2.0, _jittered(p.sw_period_channels, p.bandpass_jitter))
    sw_amp = _jittered(p.sw_amp, p.bandpass_jitter)
    sw_phase = np.random.uniform(0, 2 * np.pi)
    standing_wave = (sw_amp / 2.0) * np.sin(2 * np.pi * channels / sw_period + sw_phase)

    bandpass = envelope * (1.0 + ripple + standing_wave)
    bandpass = np.maximum(bandpass, p.bandpass_floor)

    drawn = {
        "bp_edge_frac": float(edge_frac),
        "bp_ripple_amp": float(ripple_amp),
        "bp_ripple_modes": int(n_modes),
        "bp_sw_period": float(sw_period),
        "bp_sw_amp": float(sw_amp),
    }
    return bandpass, drawn


# ============================================================================
# Section 3: RFI injection — six morphologies
#
# Every injector adds `snr * sky_sigma[f] * envelope` into `rfi_layer` and marks
# the hard geometric footprint in `injected`. `injected` is bookkeeping only: it
# records what was PUT IN so the generator can report what the bandpass later
# took away. The published label is decided in `generate_spectrogram`.
# ============================================================================

def inject_narrowband(rfi_layer, injected, n_freq, n_time, sky_sigma, n_lines, p):
    """
    Type 1 — constant narrowband: thin horizontal lines spanning all time.
    Source: cell towers, FM, aviation, GNSS — always-on fixed-frequency emitters.
    """
    for _ in range(n_lines):
        f_center = int(np.random.randint(0, n_freq))
        width = _randint(_scale_range(*p.nb_width_channels, n=n_freq))
        snr = _log_uniform(p.nb_snr)

        f_start = max(0, f_center - width // 2)
        f_end = min(n_freq, f_center + width // 2 + 1)

        for f in range(f_start, f_end):
            dist = abs(f - f_center) / max(width / 2, 1)
            envelope = np.exp(-0.5 * dist ** 2)
            intensity = snr * sky_sigma[f] * envelope
            # Real transmitters are not perfectly steady in time.
            fluctuation = 1.0 + np.random.normal(0, p.nb_time_jitter, n_time)
            rfi_layer[f, :] += intensity * np.maximum(fluctuation, 0)
            injected[f, :] = 1
    return rfi_layer, injected


def inject_wideband_burst(rfi_layer, injected, n_freq, n_time, sky_sigma, n_bursts, p):
    """
    Type 2 — instantaneous wideband: vertical columns, 1-3 time samples.
    Source: radar pulses, lightning, spark discharge, switching transients.
    """
    for _ in range(n_bursts):
        t_center = int(np.random.randint(0, n_time))
        t_width = _randint(_scale_range(*p.wb_duration_bins, n=n_time))
        f_span = int(_u(p.wb_freq_span_frac) * n_freq)
        f_start = int(np.random.randint(0, max(1, n_freq - f_span)))
        f_end = min(n_freq, f_start + f_span)
        snr = _log_uniform(p.wb_snr)

        t_start = max(0, t_center - t_width // 2)
        t_end = min(n_time, t_center + t_width // 2 + 1)

        for f in range(f_start, f_end):
            modulation = 1.0 + np.random.normal(0, p.wb_freq_jitter)
            intensity = snr * sky_sigma[f] * max(modulation, p.wb_mod_floor)
            rfi_layer[f, t_start:t_end] += intensity
            injected[f, t_start:t_end] = 1
    return rfi_layer, injected


def inject_broadband_block(rfi_layer, injected, n_freq, n_time, sky_sigma, n_blocks, p):
    """
    Type 3 — broadband blocks: rectangular patches with internal structure.
    Source: satellite passes, Wi-Fi, switch-mode power supplies.
    """
    for _ in range(n_blocks):
        lo, hi = _scale_range(*p.bb_width_channels, n=n_freq)
        f_width = _randint((lo, min(hi, max(lo + 1, n_freq // 2))))
        f_start = int(np.random.randint(0, max(1, n_freq - f_width)))
        f_end = min(n_freq, f_start + f_width)

        t_width = max(p.bb_min_time_bins, int(_u(p.bb_time_frac) * n_time))
        t_start = int(np.random.randint(0, max(1, n_time - t_width)))
        t_end = min(n_time, t_start + t_width)

        texture = np.random.normal(1.0, p.bb_texture_sd,
                                   size=(f_end - f_start, t_end - t_start))
        texture = np.maximum(gaussian_filter(texture, sigma=p.bb_texture_smooth),
                             p.bb_texture_floor)

        f_offsets = np.arange(f_end - f_start) - (f_end - f_start) / 2
        env_width = max((f_end - f_start) / p.bb_envelope_frac, 1)
        spectral_envelope = np.exp(-0.5 * (f_offsets / env_width) ** 2)

        snr = _log_uniform(p.bb_snr)
        for fi, f in enumerate(range(f_start, f_end)):
            intensity = snr * sky_sigma[f] * spectral_envelope[fi]
            rfi_layer[f, t_start:t_end] += intensity * texture[fi, :]
            injected[f, t_start:t_end] = 1
    return rfi_layer, injected


def inject_persistent_band(rfi_layer, injected, n_freq, n_time, sky_sigma, n_bands, p):
    """
    Type 4 — constant broadband: wide horizontal bands across all time.
    Source: GNSS downlinks (GPS L2, GLONASS, BeiDou). The most prominent RFI in
    real L-band data, which is why half the bands are placed in that region.
    """
    for _ in range(n_bands):
        lo, hi = _scale_range(*p.pb_width_channels, n=n_freq)
        band_width = _randint((lo, min(hi, max(lo + 1, n_freq // 4))))

        if np.random.random() < p.pb_satellite_prob:
            band_center = int(_u(p.pb_satellite_band_frac) * n_freq)
        else:
            band_center = int(np.random.randint(band_width, n_freq - band_width))

        band_start = max(0, band_center - band_width // 2)
        band_end = min(n_freq, band_center + band_width // 2 + 1)
        snr = _log_uniform(p.pb_snr)

        for f in range(band_start, band_end):
            dist = abs(f - band_center) / max(band_width / 2, 1)
            envelope = np.exp(-0.5 * (dist * p.pb_envelope_sharpness) ** 2)
            intensity = snr * sky_sigma[f] * envelope
            time_var = 1.0 + np.random.normal(0, p.pb_time_jitter, n_time)
            rfi_layer[f, :] += intensity * np.maximum(time_var, 0)
            injected[f, :] = 1
    return rfi_layer, injected


def inject_scattered(rfi_layer, injected, n_freq, n_time, sky_sigma, n_events, p):
    """
    Type 5 — occasional instantaneous narrowband: isolated dots and tiny clusters.
    Source: sporadic electronics, short transmissions, cosmic ray hits.
    """
    for _ in range(n_events):
        f_center = int(np.random.randint(0, n_freq))
        t_center = int(np.random.randint(0, n_time))
        cluster_f = _randint(_scale_range(*p.sc_cluster_channels, n=n_freq))
        cluster_t = _randint(_scale_range(*p.sc_cluster_bins, n=n_time))
        snr = _log_uniform(p.sc_snr)

        f_start = max(0, f_center - cluster_f // 2)
        f_end = min(n_freq, f_center + cluster_f // 2 + 1)
        t_start = max(0, t_center - cluster_t // 2)
        t_end = min(n_time, t_center + cluster_t // 2 + 1)

        for f in range(f_start, f_end):
            rfi_layer[f, t_start:t_end] += snr * sky_sigma[f]
            injected[f, t_start:t_end] = 1
    return rfi_layer, injected


def inject_blob(rfi_layer, injected, n_freq, n_time, sky_sigma, n_blobs, p):
    """
    Type 6 — transient broadband blobs: irregular patches bounded in BOTH axes.
    Source: satellite sidelobe illumination, intermittent broadband emitters.

    The nested loop is kept from v3 rather than vectorised: it draws no random
    numbers, so it costs nothing in reproducibility terms, and keeping it
    identical makes the v3->v4 diff auditable.
    """
    for _ in range(n_blobs):
        blob_f = _randint(_scale_range(*p.bl_size_channels, n=n_freq))
        blob_t = _randint(_scale_range(*p.bl_size_bins, n=n_time))

        f_center = int(np.random.randint(blob_f, n_freq - blob_f))
        t_center = int(np.random.randint(blob_t, n_time - blob_t))
        snr = _log_uniform(p.bl_snr)

        yy, xx = np.mgrid[-blob_f:blob_f + 1, -blob_t:blob_t + 1]

        # Rotated ellipse gives a non-rectangular outline; thresholding the soft
        # envelope roughens the boundary so blobs are not smooth ovals either.
        a = _u(p.bl_axis_frac) * blob_f
        b = _u(p.bl_axis_frac) * blob_t
        theta = np.random.uniform(0, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xx_rot = cos_t * xx + sin_t * yy
        yy_rot = -sin_t * xx + cos_t * yy
        ellipse_dist = (xx_rot / max(b, 1)) ** 2 + (yy_rot / max(a, 1)) ** 2

        blob_envelope = np.exp(-0.5 * ellipse_dist)
        blob_shape = blob_envelope > _u(p.bl_irregularity)

        texture = np.random.normal(1.0, p.bl_texture_sd, size=blob_envelope.shape)
        texture = np.maximum(gaussian_filter(texture, sigma=p.bl_texture_smooth),
                             p.bl_texture_floor)

        for dy in range(-blob_f, blob_f + 1):
            for dx in range(-blob_t, blob_t + 1):
                ly, lx = dy + blob_f, dx + blob_t
                if not blob_shape[ly, lx]:
                    continue
                f_idx, t_idx = f_center + dy, t_center + dx
                if 0 <= f_idx < n_freq and 0 <= t_idx < n_time:
                    rfi_layer[f_idx, t_idx] += (snr * sky_sigma[f_idx]
                                                * blob_envelope[ly, lx] * texture[ly, lx])
                    injected[f_idx, t_idx] = 1
    return rfi_layer, injected


# ============================================================================
# Section 4: Full image pipeline
# ============================================================================

def generate_spectrogram(n_freq, n_time, p):
    """
    One synthetic spectrogram with its bandpass, mask and strength map.

    Returns:
        spectrogram (n_freq, n_time) float32 — what the instrument records
        mask        (n_freq, n_time) uint8   — 1 where RFI is above threshold
        strength    (n_freq, n_time) float16 — post-bandpass RFI amplitude / local noise
        bandpass    (n_freq,)        float32 — the gain applied to this image
        metadata    dict
        excl        dict of per-channel injected / excluded counts (int64 arrays)
    """
    # --- density tier ---------------------------------------------------------
    r = np.random.random()
    cumulative = np.cumsum(p.density_probs)
    tier_idx = int(np.searchsorted(cumulative, r * cumulative[-1], side="right"))
    density = DENSITY_TIERS[min(tier_idx, len(DENSITY_TIERS) - 1)]

    # --- sky + noise, before the receiver ------------------------------------
    pure_signal, sky_sigma, sky_sigma_eff = generate_pure_signal(n_freq, n_time, p)

    # --- receiver gain, drawn fresh for this image ---------------------------
    bandpass, bp_drawn = generate_bandpass(n_freq, p)

    # --- RFI injection --------------------------------------------------------
    rfi_layer = np.zeros((n_freq, n_time), dtype=np.float64)
    injected = np.zeros((n_freq, n_time), dtype=np.uint8)

    metadata = {"rfi_density": density, "rfi_types": []}

    # Each type's count is an independent Poisson draw at every tier, including
    # zero. v3 changed to this from a design where type presence was tied to the
    # tier, which let a model learn "many co-occurring types = heavy" as a
    # shortcut and never produced a lone radar pulse on a quiet background.
    injectors = [
        ("narrowband", p.lam_narrowband, p.cap_narrowband, inject_narrowband),
        ("wideband_burst", p.lam_wideband, p.cap_wideband, inject_wideband_burst),
        ("broadband_block", p.lam_broadband, p.cap_broadband, inject_broadband_block),
        ("persistent_band", p.lam_persistent, p.cap_persistent, inject_persistent_band),
        ("blob", p.lam_blob, p.cap_blob, inject_blob),
        ("scattered", p.lam_scattered, p.cap_scattered, inject_scattered),
    ]

    for name, lambdas, cap, fn in injectors:
        lam = lambdas[tier_idx]
        if lam <= 0:
            continue
        count = min(int(np.random.poisson(lam=lam)), cap)
        if count > 0:
            rfi_layer, injected = fn(rfi_layer, injected, n_freq, n_time,
                                     sky_sigma, count, p)
            metadata["rfi_types"].append(f"{name} x{count}")

    # Spectral leakage smears RFI power into neighbouring pixels.
    if np.any(rfi_layer > 0):
        rfi_layer = gaussian_filter(rfi_layer, sigma=_u(p.rfi_blur))

    # --- the receiver chain ---------------------------------------------------
    # One multiplicative gain on everything that reaches the receiver — sky,
    # noise and RFI alike — then a fixed additive noise floor from the amplifier
    # and digitiser, which enters AFTER the gain and so does not scale with it.
    sigma_rx = p.rx_noise_frac * float(np.median(sky_sigma_eff))
    rx_noise = (np.random.normal(0, sigma_rx, size=(n_freq, n_time))
                if sigma_rx > 0 else 0.0)

    spectrogram = bandpass[:, np.newaxis] * (pure_signal + rfi_layer) + rx_noise

    # RFI as it actually appears at the output, and the noise it competes with.
    rfi_observed = bandpass[:, np.newaxis] * rfi_layer
    sigma_total = np.sqrt((bandpass * sky_sigma_eff) ** 2 + sigma_rx ** 2)

    # --- the label ------------------------------------------------------------
    # Threshold the POST-bandpass amplitude against the LOCAL total noise. Where
    # the gain has buried the RFI below the noise there is nothing left to
    # detect, so it is not labelled — otherwise the labels would be unsatisfiable
    # and every score computed against them would overstate what a model failed
    # to do.
    # Cast to float16 BEFORE thresholding, not after, so that the saved mask is
    # exactly reproducible from the saved strength map. Thresholding at float64
    # and then narrowing would let boundary pixels disagree on disk.
    strength = (rfi_observed / sigma_total[:, np.newaxis]).astype(np.float16)
    mask = (strength > p.mask_sigma_threshold).astype(np.uint8)

    # --- what the bandpass cost, per frequency channel ------------------------
    injected_bool = injected.astype(bool)
    excluded_bool = injected_bool & (mask == 0)
    excl = {
        "injected": injected_bool.sum(axis=1).astype(np.int64),
        "excluded": excluded_bool.sum(axis=1).astype(np.int64),
    }

    # --- metadata -------------------------------------------------------------
    n_injected = int(injected_bool.sum())
    n_excluded = int(excluded_bool.sum())
    metadata["rfi_fraction"] = float(mask.sum() / mask.size)
    metadata["n_freq"] = n_freq
    metadata["n_time"] = n_time
    metadata["n_injected_px"] = n_injected
    metadata["n_labelled_px"] = int(mask.sum())
    metadata["n_excluded_px"] = n_excluded
    metadata["frac_injected_excluded"] = float(n_excluded / n_injected) if n_injected else 0.0
    metadata["sigma_rx"] = float(sigma_rx)
    metadata["bandpass_min"] = float(bandpass.min())
    metadata["bandpass_max"] = float(bandpass.max())
    metadata["bandpass_mean"] = float(bandpass.mean())
    metadata.update(bp_drawn)

    m = mask.astype(bool)
    if m.sum() > 0:
        s = strength[m].astype(np.float32)
        metadata["strength_median"] = float(np.median(s))
        metadata["strength_p10"] = float(np.percentile(s, 10))
        metadata["strength_p90"] = float(np.percentile(s, 90))
        metadata["frac_weak"] = float((s < p.strength_weak).mean())
        metadata["frac_medium"] = float(((s >= p.strength_weak) & (s < p.strength_strong)).mean())
        metadata["frac_strong"] = float((s >= p.strength_strong).mean())
    else:
        for k in ("strength_median", "strength_p10", "strength_p90",
                  "frac_weak", "frac_medium", "frac_strong"):
            metadata[k] = 0.0

    return (spectrogram.astype(np.float32), mask, strength,
            bandpass.astype(np.float32), metadata, excl)


# ============================================================================
# Section 5: Preview
# ============================================================================

def plot_preview(spectrogram, mask, bandpass, metadata, filename):
    """Four panels: what was recorded, the label, the label applied, and B(f)."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    im0 = axes[0].imshow(spectrogram, aspect="auto", cmap="hot", origin="upper")
    axes[0].set_title("Spectrogram (bandpass applied)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time sample"); axes[0].set_ylabel("Frequency channel")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mask, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="upper")
    axes[1].set_title(f"Ground truth ({metadata['rfi_fraction']*100:.1f}% RFI)",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time sample"); axes[1].set_ylabel("Frequency channel")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    palette = copy(plt.cm.hot)
    palette.set_bad("cyan", 1.0)
    masked = np.ma.array(spectrogram, mask=mask)
    vmax = (np.percentile(spectrogram[mask == 0], 98)
            if np.any(mask == 0) else float(spectrogram.max()))
    im2 = axes[2].imshow(masked, aspect="auto", cmap=palette,
                         vmin=float(spectrogram.min()), vmax=vmax, origin="upper")
    axes[2].set_title(f"Cleaned — {metadata['rfi_density']}", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time sample"); axes[2].set_ylabel("Frequency channel")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].plot(bandpass, np.arange(len(bandpass)), lw=1.2, color="#1f77b4")
    axes[3].set_ylim(len(bandpass) - 1, 0)          # match imshow's origin='upper'
    axes[3].set_xlim(0, max(1.05, float(bandpass.max()) * 1.05))
    axes[3].set_title(f"Bandpass B(f)  —  {metadata['n_excluded_px']:,} px excluded",
                      fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Gain"); axes[3].set_ylabel("Frequency channel")
    axes[3].grid(alpha=0.3)

    types = ", ".join(metadata["rfi_types"]) if metadata["rfi_types"] else "None (clean)"
    fig.suptitle(f"RFI types: {types}", fontsize=10, y=0.02, color="gray")
    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Section 6: Dataset loop
# ============================================================================

def generate_dataset(p):
    n_freq, n_time, n_images = p.n_freq, p.n_time, p.n_images

    n_train = int(p.train_frac * n_images)
    n_val = int(p.val_frac * n_images)
    splits = {"train": n_train, "val": n_val, "test": n_images - n_train - n_val}

    for split in splits:
        for sub in ("images", "masks", "strength", "bandpass"):
            os.makedirs(os.path.join(p.output_dir, split, sub), exist_ok=True)
    os.makedirs(os.path.join(p.output_dir, "preview"), exist_ok=True)

    print("=" * 74)
    print("Synthetic RFI Dataset Generator v4 — instrument bandpass")
    print("=" * 74)
    print(f"  Image shape (freq x time) : {n_freq} x {n_time}  ({n_freq*n_time:,} px)")
    print(f"  Images                    : {n_images}  "
          f"(train {splits['train']} / val {splits['val']} / test {splits['test']})")
    print(f"  Seed                      : {p.seed}")
    print(f"  Bandpass edge_frac        : {p.edge_frac}   jitter {p.bandpass_jitter}")
    print(f"  Ripple / standing wave    : {p.ripple_amp} p-p / {p.sw_amp} @ "
          f"{p.sw_period_channels} ch")
    print(f"  Post-gain noise floor     : {p.rx_noise_frac} x median sky sigma")
    print(f"  Label rule                : post-bandpass RFI > "
          f"{p.mask_sigma_threshold} x local total sigma")
    print(f"  Output                    : {p.output_dir}")
    print("=" * 74)

    stats = {t: 0 for t in DENSITY_TIERS}
    stats.update({"rfi_fractions": [], "rfi_type_counts": {},
                  "injected_px": 0, "excluded_px": 0})
    per_channel = {s: {"injected": np.zeros(n_freq, dtype=np.int64),
                       "excluded": np.zeros(n_freq, dtype=np.int64)} for s in splits}

    global_idx = 0
    for split_name, split_count in splits.items():
        print(f"\nGenerating {split_name} ({split_count} images)...")
        meta_path = os.path.join(p.output_dir, split_name, "metadata.jsonl")

        with open(meta_path, "w") as meta_f:
            for i in range(split_count):
                spec, mask, strength, bandpass, md, excl = generate_spectrogram(
                    n_freq, n_time, p)

                base = os.path.join(p.output_dir, split_name)
                np.save(os.path.join(base, "images", f"spectrogram_{global_idx:04d}.npy"), spec)
                np.save(os.path.join(base, "masks", f"mask_{global_idx:04d}.npy"), mask)
                np.save(os.path.join(base, "strength", f"strength_{global_idx:04d}.npy"), strength)
                np.save(os.path.join(base, "bandpass", f"bandpass_{global_idx:04d}.npy"), bandpass)

                if global_idx < p.n_previews:
                    plot_preview(spec, mask, bandpass, md,
                                 os.path.join(p.output_dir, "preview",
                                              f"preview_{global_idx:04d}.png"))

                # The first 11 fields are exactly v3's, in v3's order, so
                # anything already reading this file keeps working.
                record = {
                    "global_idx": global_idx,
                    "split": split_name,
                    "rfi_density": md["rfi_density"],
                    "rfi_types": md["rfi_types"],
                    "rfi_fraction": md["rfi_fraction"],
                    "strength_median": md["strength_median"],
                    "strength_p10": md["strength_p10"],
                    "strength_p90": md["strength_p90"],
                    "frac_weak": md["frac_weak"],
                    "frac_medium": md["frac_medium"],
                    "frac_strong": md["frac_strong"],
                }
                for k in ("n_injected_px", "n_labelled_px", "n_excluded_px",
                          "frac_injected_excluded", "sigma_rx",
                          "bandpass_min", "bandpass_max", "bandpass_mean",
                          "bp_edge_frac", "bp_ripple_amp", "bp_ripple_modes",
                          "bp_sw_period", "bp_sw_amp"):
                    record[k] = md[k]
                meta_f.write(json.dumps(record) + "\n")

                stats[md["rfi_density"]] += 1
                stats["rfi_fractions"].append(md["rfi_fraction"])
                stats["injected_px"] += md["n_injected_px"]
                stats["excluded_px"] += md["n_excluded_px"]
                for t in md["rfi_types"]:
                    name = t.split(" x")[0]
                    stats["rfi_type_counts"][name] = stats["rfi_type_counts"].get(name, 0) + 1
                per_channel[split_name]["injected"] += excl["injected"]
                per_channel[split_name]["excluded"] += excl["excluded"]

                global_idx += 1
                if (i + 1) % 50 == 0 or (i + 1) == split_count:
                    print(f"  [{split_name}] {i+1}/{split_count}")

    _write_reports(p, splits, stats, per_channel, n_freq)
    return stats


def _write_reports(p, splits, stats, per_channel, n_freq):
    """Summary to stdout, plus dataset_statistics.txt and exclusion_by_channel.json."""
    n_images = p.n_images
    fracs = np.array(stats["rfi_fractions"])

    print(f"\n{'=' * 74}\nGeneration complete\n{'=' * 74}")
    print("\nRFI density distribution:")
    for t in DENSITY_TIERS:
        print(f"  {t:>10s}: {stats[t]:4d} images ({100*stats[t]/n_images:5.1f}%)")

    print("\nRFI type frequency:")
    for t, c in sorted(stats["rfi_type_counts"].items()):
        print(f"  {t:>20s}: {c:4d} occurrences")

    print("\nRFI pixel fraction (labelled):")
    print(f"  min {100*fracs.min():6.2f}%   median {100*np.median(fracs):6.2f}%   "
          f"mean {100*fracs.mean():6.2f}%   max {100*fracs.max():6.2f}%")

    # What the bandpass cost.
    inj = sum(per_channel[s]["injected"] for s in splits)
    exc = sum(per_channel[s]["excluded"] for s in splits)
    tot_inj, tot_exc = int(inj.sum()), int(exc.sum())
    print(f"\n{'-' * 74}\nWHAT THE BANDPASS COST (injected RFI pixels dropped below "
          f"{p.mask_sigma_threshold} sigma)\n{'-' * 74}")
    print(f"  injected : {tot_inj:,}")
    print(f"  excluded : {tot_exc:,}  ({100*tot_exc/max(tot_inj,1):.3f}%)")

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(inj > 0, exc / np.maximum(inj, 1), 0.0)

    # Which channels are actually being destroyed, and how far in it goes.
    heavy = np.where(rate > p.report_channel_loss_threshold)[0]
    if heavy.size:
        left = heavy[heavy < n_freq // 2]
        right = heavy[heavy >= n_freq // 2]
        print(f"\n  channels losing >{100*p.report_channel_loss_threshold:.0f}% of "
              f"injected RFI: {heavy.size} of {n_freq} ({100*heavy.size/n_freq:.2f}%)")
        if left.size:
            print(f"    low-frequency edge : channels 0-{left.max()} "
                  f"({left.size} channels)")
        if right.size:
            print(f"    high-frequency edge: channels {right.min()}-{n_freq-1} "
                  f"({right.size} channels)")
    else:
        print(f"\n  no channel loses more than "
              f"{100*p.report_channel_loss_threshold:.0f}% of its injected RFI")

    # Deciles across the band make the shape of the loss visible at a glance.
    print("\n  exclusion rate by band position (decile):")
    edges = np.linspace(0, n_freq, 11).astype(int)
    for k in range(10):
        a, b = edges[k], edges[k + 1]
        di, de = int(inj[a:b].sum()), int(exc[a:b].sum())
        r = 100 * de / di if di else 0.0
        print(f"    ch {a:5d}-{b-1:5d} : {r:6.2f}%   ({de:,} of {di:,})")

    stats_path = os.path.join(p.output_dir, "dataset_statistics.txt")
    with open(stats_path, "w") as f:
        f.write("Synthetic RFI Dataset v4 — instrument bandpass\n")
        f.write("=" * 60 + "\n")
        f.write(f"Random seed: {p.seed}\n")
        f.write(f"Image size (freq x time): {p.n_freq} x {p.n_time}\n")
        f.write(f"Total images: {n_images}\n")
        f.write(f"Train: {splits['train']}, Val: {splits['val']}, Test: {splits['test']}\n\n")
        f.write("Bandpass: edge_frac=%g jitter=%g ripple_amp=%g sw_period=%g sw_amp=%g\n"
                % (p.edge_frac, p.bandpass_jitter, p.ripple_amp,
                   p.sw_period_channels, p.sw_amp))
        f.write("Post-gain receiver noise: %g x median sky sigma\n" % p.rx_noise_frac)
        f.write("Label: post-bandpass RFI > %g x local total sigma\n\n"
                % p.mask_sigma_threshold)
        f.write("RFI density distribution:\n")
        for t in DENSITY_TIERS:
            f.write(f"  {t}: {stats[t]} ({100*stats[t]/n_images:.1f}%)\n")
        f.write("\nRFI type frequency:\n")
        for t, c in sorted(stats["rfi_type_counts"].items()):
            f.write(f"  {t}: {c}\n")
        f.write(f"\nRFI pixel fraction: min={100*fracs.min():.2f}%, "
                f"median={100*np.median(fracs):.2f}%, mean={100*fracs.mean():.2f}%, "
                f"max={100*fracs.max():.2f}%\n")
        f.write(f"\nInjected RFI pixels: {tot_inj}\n")
        f.write(f"Excluded by bandpass: {tot_exc} ({100*tot_exc/max(tot_inj,1):.3f}%)\n")

    excl_path = os.path.join(p.output_dir, "exclusion_by_channel.json")
    with open(excl_path, "w") as f:
        json.dump({
            "n_freq": n_freq,
            "mask_sigma_threshold": p.mask_sigma_threshold,
            "total": {"injected": inj.tolist(), "excluded": exc.tolist()},
            "by_split": {s: {"injected": per_channel[s]["injected"].tolist(),
                             "excluded": per_channel[s]["excluded"].tolist()}
                         for s in splits},
        }, f)

    print(f"\nStatistics : {stats_path}")
    print(f"Per-channel: {excl_path}")
    print(f"Previews   : {os.path.join(p.output_dir, 'preview')}")
    print("\nDone.")


# ============================================================================
# Section 7: CLI — every constant above is a flag, no bare literals
# ============================================================================

def build_parser():
    ap = argparse.ArgumentParser(
        description="Generate synthetic RFI spectrograms with a per-image instrument bandpass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = ap.add_argument_group("dimensions, splits, output")
    g.add_argument("--n_freq", type=int, default=1024, help="frequency channels (axis 0)")
    g.add_argument("--n_time", type=int, default=265, help="time samples (axis 1)")
    g.add_argument("--n_images", type=int, default=1000, help="total images across all splits")
    g.add_argument("--train_frac", type=float, default=0.70, help="fraction to train split")
    g.add_argument("--val_frac", type=float, default=0.15, help="fraction to val split (test gets the remainder)")
    g.add_argument("--n_previews", type=int, default=12, help="preview PNGs to render")
    g.add_argument("--output_dir", type=str, required=True, help="dataset root directory")
    g.add_argument("--seed", type=int, default=42, help="RNG seed; the dataset regenerates bit-exactly from it")

    g = ap.add_argument_group("bandpass B(f) — the new physics")
    g.add_argument("--edge_frac", type=float, default=0.08,
                   help="fraction of the band in Tukey roll-off at EACH edge")
    g.add_argument("--ripple_amp", type=float, default=0.15,
                   help="slow ripple, fractional peak-to-peak (worst case, all modes in phase)")
    g.add_argument("--ripple_modes", type=int, nargs=2, default=[2, 4],
                   help="number of ripple sinusoids, randint(lo, hi)")
    g.add_argument("--ripple_period_frac", type=float, nargs=2, default=[0.4, 1.5],
                   help="ripple period as a fraction of the band")
    g.add_argument("--sw_period_channels", type=float, default=128.0,
                   help="standing-wave period in frequency channels")
    g.add_argument("--sw_amp", type=float, default=0.05,
                   help="standing-wave fractional peak-to-peak amplitude")
    g.add_argument("--bandpass_jitter", type=float, default=0.25,
                   help="how much per-image draws differ; each amplitude is scaled by "
                        "U(1-j, 1+j). At 0 the amplitudes match but phases still vary.")
    g.add_argument("--bandpass_floor", type=float, default=1e-3,
                   help="minimum gain, so no channel is identically dead")

    g = ap.add_argument_group("receiver noise floor (post-gain)")
    g.add_argument("--rx_noise_frac", type=float, default=0.10,
                   help="amplifier/digitiser noise sd as a fraction of the median sky "
                        "noise sd. Enters AFTER the gain, so it does not scale with it — "
                        "this is what lets the bandpass bury edge-channel RFI. Set 0 for "
                        "the pure-radiometer model where detectability is gain-independent.")

    g = ap.add_argument_group("sky pedestal and background noise")
    g.add_argument("--freq_lo_mhz", type=float, default=1000.0)
    g.add_argument("--freq_hi_mhz", type=float, default=1500.0)
    g.add_argument("--sed_center_mhz", type=float, nargs=2, default=[1200.0, 1300.0])
    g.add_argument("--sed_width_mhz", type=float, nargs=2, default=[150.0, 250.0])
    g.add_argument("--sed_amp", type=float, nargs=2, default=[8.0, 15.0])
    g.add_argument("--sed_ripple_modes", type=int, nargs=2, default=[3, 8])
    g.add_argument("--sed_ripple_frac", type=float, nargs=2, default=[0.02, 0.08])
    g.add_argument("--sed_floor", type=float, default=0.1)
    g.add_argument("--sigma_bg", type=float, nargs=2, default=[2.0, 6.0],
                   help="background noise sd, uniform draw per image")
    g.add_argument("--hetero_floor", type=float, default=0.05,
                   help="floor on normalised pedestal when scaling noise, caps sigma at low gain")
    g.add_argument("--corr_sigma_freq", type=float, nargs=2, default=[0.3, 0.8],
                   help="channel-to-channel correlation, Gaussian sigma")
    g.add_argument("--corr_sigma_time", type=float, nargs=2, default=[0.3, 0.3],
                   help="time correlation, Gaussian sigma")
    g.add_argument("--drift_amp", type=float, nargs=2, default=[0.0, 0.05])
    g.add_argument("--drift_modes", type=int, nargs=2, default=[1, 4])
    g.add_argument("--drift_period_frac", type=float, nargs=2, default=[0.3, 2.0])
    g.add_argument("--v3_edge_rolloff", action="store_true",
                   help="also apply v3's linear edge rolloff. OFF by default: the Tukey "
                        "taper supersedes it and stacking both attenuates edges twice.")
    g.add_argument("--v3_rolloff_channels", type=int, nargs=2, default=[20, 60])
    g.add_argument("--v3_rolloff_min", type=float, default=0.05)

    g = ap.add_argument_group("RFI density tiers")
    g.add_argument("--density_probs", type=float, nargs=4, default=[0.08, 0.29, 0.38, 0.25],
                   help="probability of clean / light / moderate / heavy")

    g = ap.add_argument_group("type 1 — narrowband (horizontal lines)")
    g.add_argument("--lam_narrowband", type=float, nargs=4, default=[0.0, 2.0, 6.0, 11.0],
                   help="Poisson lambda per density tier")
    g.add_argument("--cap_narrowband", type=int, default=20)
    g.add_argument("--nb_width_channels", type=int, nargs=2, default=[1, 6],
                   help="line width in channels, quoted at the 1024 reference")
    g.add_argument("--nb_snr", type=float, nargs=2, default=[1.5, 12.0],
                   help="log-uniform SNR relative to local noise")
    g.add_argument("--nb_time_jitter", type=float, default=0.05)

    g = ap.add_argument_group("type 2 — wideband burst (vertical columns)")
    g.add_argument("--lam_wideband", type=float, nargs=4, default=[0.0, 0.05, 0.3, 1.5])
    g.add_argument("--cap_wideband", type=int, default=8)
    g.add_argument("--wb_duration_bins", type=int, nargs=2, default=[1, 4])
    g.add_argument("--wb_freq_span_frac", type=float, nargs=2, default=[0.5, 1.0])
    g.add_argument("--wb_snr", type=float, nargs=2, default=[3.0, 20.0])
    g.add_argument("--wb_freq_jitter", type=float, default=0.1)
    g.add_argument("--wb_mod_floor", type=float, default=0.1)

    g = ap.add_argument_group("type 3 — broadband block")
    g.add_argument("--lam_broadband", type=float, nargs=4, default=[0.0, 0.15, 1.0, 2.2])
    g.add_argument("--cap_broadband", type=int, default=5)
    g.add_argument("--bb_width_channels", type=int, nargs=2, default=[30, 301])
    g.add_argument("--bb_time_frac", type=float, nargs=2, default=[0.1, 0.8])
    g.add_argument("--bb_min_time_bins", type=int, default=5)
    g.add_argument("--bb_snr", type=float, nargs=2, default=[1.5, 8.0])
    g.add_argument("--bb_texture_sd", type=float, default=0.2)
    g.add_argument("--bb_texture_smooth", type=float, default=3.0)
    g.add_argument("--bb_texture_floor", type=float, default=0.1)
    g.add_argument("--bb_envelope_frac", type=float, default=3.0)

    g = ap.add_argument_group("type 4 — persistent band")
    g.add_argument("--lam_persistent", type=float, nargs=4, default=[0.0, 0.25, 1.5, 3.0])
    g.add_argument("--cap_persistent", type=int, default=6)
    g.add_argument("--pb_width_channels", type=int, nargs=2, default=[10, 101])
    g.add_argument("--pb_satellite_prob", type=float, default=0.5)
    g.add_argument("--pb_satellite_band_frac", type=float, nargs=2, default=[0.32, 0.60],
                   help="band position as a fraction of the axis (~1160-1300 MHz GNSS region)")
    g.add_argument("--pb_snr", type=float, nargs=2, default=[2.0, 15.0])
    g.add_argument("--pb_envelope_sharpness", type=float, default=1.5)
    g.add_argument("--pb_time_jitter", type=float, default=0.03)

    g = ap.add_argument_group("type 5 — scattered dots")
    g.add_argument("--lam_scattered", type=float, nargs=4, default=[0.0, 3.0, 10.0, 25.0])
    g.add_argument("--cap_scattered", type=int, default=80)
    g.add_argument("--sc_cluster_channels", type=int, nargs=2, default=[1, 6])
    g.add_argument("--sc_cluster_bins", type=int, nargs=2, default=[1, 6])
    g.add_argument("--sc_snr", type=float, nargs=2, default=[2.0, 8.0])

    g = ap.add_argument_group("type 6 — blob")
    g.add_argument("--lam_blob", type=float, nargs=4, default=[0.0, 0.4, 2.0, 5.0])
    g.add_argument("--cap_blob", type=int, default=15)
    g.add_argument("--bl_size_channels", type=int, nargs=2, default=[5, 40])
    g.add_argument("--bl_size_bins", type=int, nargs=2, default=[3, 20])
    g.add_argument("--bl_snr", type=float, nargs=2, default=[2.0, 12.0])
    g.add_argument("--bl_axis_frac", type=float, nargs=2, default=[0.6, 1.0])
    g.add_argument("--bl_irregularity", type=float, nargs=2, default=[0.3, 0.6])
    g.add_argument("--bl_texture_sd", type=float, default=0.25)
    g.add_argument("--bl_texture_smooth", type=float, default=1.5)
    g.add_argument("--bl_texture_floor", type=float, default=0.1)

    g = ap.add_argument_group("labelling and reporting")
    g.add_argument("--rfi_blur", type=float, nargs=2, default=[0.3, 0.7],
                   help="Gaussian sigma applied to the RFI layer — spectral leakage")
    g.add_argument("--mask_sigma_threshold", type=float, default=0.5,
                   help="a pixel is RFI when post-bandpass amplitude exceeds this many "
                        "local total noise sigma. v3 used 0.5 pre-bandpass.")
    g.add_argument("--strength_weak", type=float, default=3.0)
    g.add_argument("--strength_strong", type=float, default=7.0)
    g.add_argument("--report_channel_loss_threshold", type=float, default=0.5,
                   help="a channel is reported as badly damaged when it loses more than "
                        "this fraction of its injected RFI pixels")
    return ap


def validate(p):
    """Fail loudly on parameter combinations that would produce a silently wrong dataset."""
    errors = []
    if p.n_freq < 2 or p.n_time < 2:
        errors.append("--n_freq and --n_time must both be >= 2")
    if not 0 < p.train_frac < 1 or not 0 < p.val_frac < 1:
        errors.append("--train_frac and --val_frac must be in (0, 1)")
    if p.train_frac + p.val_frac >= 1:
        errors.append("--train_frac + --val_frac must leave room for a test split")
    if not 0 <= p.edge_frac < 0.5:
        errors.append("--edge_frac must be in [0, 0.5)")
    if p.bandpass_jitter < 0:
        errors.append("--bandpass_jitter must be >= 0")
    if p.rx_noise_frac < 0:
        errors.append("--rx_noise_frac must be >= 0")
    if p.bandpass_floor <= 0:
        errors.append("--bandpass_floor must be > 0 (it guards a division)")
    if any(d < 0 for d in p.density_probs) or sum(p.density_probs) <= 0:
        errors.append("--density_probs must be non-negative and sum to > 0")
    if p.mask_sigma_threshold < 0:
        errors.append("--mask_sigma_threshold must be >= 0")
    # A blob is placed with randint(size, n - size), which needs n > 2*size.
    bl_f_hi = _scale_range(*p.bl_size_channels, n=p.n_freq)[1]
    bl_t_hi = _scale_range(*p.bl_size_bins, n=p.n_time)[1]
    if p.n_freq <= 2 * bl_f_hi:
        errors.append(f"--n_freq {p.n_freq} too small for blobs up to {bl_f_hi} channels")
    if p.n_time <= 2 * bl_t_hi:
        errors.append(f"--n_time {p.n_time} too small for blobs up to {bl_t_hi} bins")
    for name in ("nb_snr", "wb_snr", "bb_snr", "pb_snr", "sc_snr", "bl_snr"):
        lo, hi = getattr(p, name)
        if lo <= 0 or hi < lo:
            errors.append(f"--{name} must satisfy 0 < lo <= hi (log-uniform draw)")
    if errors:
        raise SystemExit("Invalid parameters:\n  - " + "\n  - ".join(errors))


if __name__ == "__main__":
    params = build_parser().parse_args()
    validate(params)

    # Seed once. Every draw after this comes from numpy's legacy global
    # RandomState in a fixed call order, which is what makes the dataset
    # regenerate bit-exactly.
    np.random.seed(params.seed)

    generate_dataset(params)
