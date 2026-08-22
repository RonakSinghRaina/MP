"""
Synthetic Radio Astronomy Spectrogram Dataset Generator
========================================================
Generates realistic 2D spectrograms with injected RFI and perfect ground-truth binary masks
for training ML models (U-Net, CNN, CRF) to detect Radio Frequency Interference.

Based on the physics and methodology described in:
- Chen et al. (2023) "Cleaning RFI in Pulsar-Folded Data Based on the CRFs with an Adaptive Prior"
- FAST telescope L-band (1000-1500 MHz) observation characteristics

Implements all 6 RFI morphology types:
  Type 1: Constant Narrowband RFI (horizontal thin lines)
  Type 2: Instantaneous Wideband RFI (vertical columns)
  Type 3: Broadband RFI (rectangular blocks)
  Type 4: Constant Broadband RFI (persistent wide horizontal bands)
  Type 5: Occasional Instantaneous Narrowband RFI (scattered dots)
  Type 6: Blob / Transient Broadband RFI (irregular-shaped short patches)

Usage:
    python dataset_generator.py                         # Generate 1000 images (default)
    python dataset_generator.py --n_images 500          # Generate 500 images
    python dataset_generator.py --n_images 200 --size 512  # 200 images at 512x512
    python dataset_generator.py --preview_only          # Generate 20 preview images only
"""

import os
import json
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from copy import copy


# ---------------------------------------------------------------------------
# Image-size scaling of RFI morphology
# ---------------------------------------------------------------------------
# All pixel-valued morphology parameters below were originally tuned for
# 1024 x 1024 images. If the image size changes (e.g. to the 276 x 600 used
# by Akeret et al. 2017), those raw pixel counts no longer represent the same
# physical structures: a 40-channel blob is 3.9% of a 1024-channel band but
# 14.5% of a 276-channel band. Scaling keeps RFI structures at constant
# FRACTIONAL size, which is what makes datasets at different resolutions
# physically comparable.
#
# NOTE: parameters expressed in MHz (e.g. the bandpass gain curve width) are
# already resolution-independent and are deliberately NOT scaled.
REFERENCE_DIM = 1024


def _scale_range(lo, hi, n, reference=REFERENCE_DIM, min_lo=1):
    """Scale an integer pixel range [lo, hi) by n / reference."""
    f = n / float(reference)
    slo = max(min_lo, int(round(lo * f)))
    shi = max(slo + 1, int(round(hi * f)))
    return slo, shi


# ============================================================================
# Section 1: Pure Signal Generation (Background Thermal Noise)
# ============================================================================

def generate_bandpass_gain_curve(n_freq, center_freq=None, bandwidth_sigma=None, amplitude=None):
    """
    Generates a realistic telescope bandpass gain curve.
    
    Real radio telescopes (like FAST) do NOT have uniform sensitivity across
    all frequencies. The receiver gain varies smoothly across the band,
    creating a characteristic spectral energy distribution (SED) curve.
    
    Parameters:
        n_freq: Number of frequency channels
        center_freq: Center frequency of gain peak in MHz (randomized if None)
        bandwidth_sigma: Width of gain curve in MHz (randomized if None)
        amplitude: Peak gain amplitude (randomized if None)
    
    Returns:
        gain_curve: 1D array of shape (n_freq,) with gain values
    """
    freqs = np.linspace(1000, 1500, n_freq)  # MHz
    
    if center_freq is None:
        center_freq = np.random.uniform(1200, 1300)
    if bandwidth_sigma is None:
        bandwidth_sigma = np.random.uniform(150, 250)
    if amplitude is None:
        amplitude = np.random.uniform(8, 15)
    
    # Gaussian-shaped bandpass gain curve
    gain_curve = amplitude * np.exp(-0.5 * ((freqs - center_freq) / bandwidth_sigma) ** 2)
    
    # Add subtle ripple structure (real receivers have small gain ripples)
    n_ripples = np.random.randint(3, 8)
    ripple_amplitude = np.random.uniform(0.02, 0.08) * amplitude
    ripple_phase = np.random.uniform(0, 2 * np.pi)
    gain_curve += ripple_amplitude * np.sin(n_ripples * np.pi * np.linspace(0, 1, n_freq) + ripple_phase)
    
    return np.maximum(gain_curve, 0.1)


def generate_edge_rolloff(n_freq, rolloff_width=None):
    """
    Simulates the characteristic dark edges at band boundaries.
    
    Real FAST data has very low gain at the edges of the 1000-1500 MHz band.
    The first/last ~50-128 channels are often removed entirely.
    
    Parameters:
        n_freq: Number of frequency channels
        rolloff_width: Width of rolloff in channels (randomized if None)
    
    Returns:
        edge_mask: 1D array of shape (n_freq,) with values in [0.05, 1.0]
    """
    if rolloff_width is None:
        _lo, _hi = _scale_range(20, 60, n_freq)
        rolloff_width = np.random.randint(_lo, _hi)
    
    edge_mask = np.ones(n_freq)
    # Bottom edge (low frequency)
    edge_mask[:rolloff_width] = np.linspace(0.05, 1.0, rolloff_width)
    # Top edge (high frequency)
    edge_mask[-rolloff_width:] = np.linspace(1.0, 0.05, rolloff_width)
    
    return edge_mask


def generate_temporal_drift(n_time, drift_amplitude=None):
    """
    Simulates slow temporal gain drift due to atmospheric/electronic changes.
    
    Over the course of an observation, telescope gain can drift slowly due to
    atmospheric conditions, telescope elevation changes, or electronics warming up.
    
    Parameters:
        n_time: Number of time bins
        drift_amplitude: Fractional amplitude of drift (randomized if None)
    
    Returns:
        drift: 1D array of shape (n_time,) with multiplicative drift factors ~1.0
    """
    if drift_amplitude is None:
        drift_amplitude = np.random.uniform(0.0, 0.05)
    
    # Low-frequency sinusoidal drift with random phase
    n_modes = np.random.randint(1, 4)
    drift = np.ones(n_time)
    for _ in range(n_modes):
        period = np.random.uniform(0.3, 2.0) * n_time
        phase = np.random.uniform(0, 2 * np.pi)
        drift += drift_amplitude / n_modes * np.sin(2 * np.pi * np.arange(n_time) / period + phase)
    
    return drift


def generate_pure_signal(n_freq, n_time, sigma_bg=None):
    """
    Generates a realistic pure thermal noise spectrogram (no RFI).
    
    Implements the radiometer equation physics:
    - Gaussian thermal noise (Johnson-Nyquist noise from receiver electronics)
    - Frequency-dependent bandpass gain curve
    - Heteroscedastic noise (noise level varies with gain)
    - Band edge rolloff
    - Temporal drift
    - Slight spatial correlation between adjacent channels
    
    NOTE: the background *mean* level is set entirely by the bandpass gain
    curve (generate_bandpass_gain_curve), not by a separate mu_bg parameter.
    An earlier version of this function sampled an unused mu_bg value that
    had no effect on the output — it has been removed to avoid confusion.
    
    Parameters:
        n_freq: Number of frequency channels (rows)
        n_time: Number of time bins (columns)
        sigma_bg: Background noise standard deviation (randomized if None)
    
    Returns:
        pure_signal: 2D array of shape (n_freq, n_time), dtype float64
        gain_curve: 1D array of shape (n_freq,) — the bandpass gain used
        local_sigma: 1D array of shape (n_freq,) — per-channel noise sigma
    """
    if sigma_bg is None:
        sigma_bg = np.random.uniform(2, 6)
    
    # 1. Bandpass gain curve (frequency-dependent sensitivity)
    gain_curve = generate_bandpass_gain_curve(n_freq)
    
    # 2. Edge rolloff
    edge_mask = generate_edge_rolloff(n_freq)
    gain_with_edges = gain_curve * edge_mask
    
    # 3. Per-channel noise sigma (heteroscedastic — noisier at low gain)
    #    sigma_ch is inversely proportional to sqrt(gain), normalized to sigma_bg at peak
    gain_normalized = gain_with_edges / np.max(gain_with_edges)
    local_sigma = sigma_bg / np.sqrt(np.maximum(gain_normalized, 0.05))
    
    # 4. Generate Gaussian thermal noise per pixel
    noise = np.random.normal(0, 1, size=(n_freq, n_time))
    pure_signal = gain_with_edges[:, np.newaxis] + local_sigma[:, np.newaxis] * noise
    
    # 5. Temporal drift
    drift = generate_temporal_drift(n_time)
    pure_signal *= drift[np.newaxis, :]
    
    # 6. Slight spatial correlation (adjacent channels are not perfectly independent)
    correlation_sigma = np.random.uniform(0.3, 0.8)
    pure_signal = gaussian_filter(pure_signal, sigma=(correlation_sigma, 0.3))
    
    return pure_signal, gain_with_edges, local_sigma


# ============================================================================
# Section 2: RFI Injection Functions (All 5 Types)
# ============================================================================

def log_uniform(low, high):
    """Sample from a log-uniform distribution (equal probability per decade)."""
    return np.exp(np.random.uniform(np.log(low), np.log(high)))


def inject_narrowband_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_lines=None):
    """
    Type 1: Constant Narrowband RFI (Horizontal Thin Lines)
    
    Physical source: Cell towers, FM radio, aviation, GPS/GLONASS/BeiDou satellites
    transmitting at fixed frequencies.
    
    Appearance: Thin horizontal lines spanning the entire time axis at fixed
    frequency channels. "Constant" because the source is always transmitting.
    """
    if n_lines is None:
        n_lines = np.random.poisson(lam=5)
    n_lines = min(n_lines, 20)  # Cap at 20
    
    for _ in range(n_lines):
        # Random frequency position
        freq_center = np.random.randint(0, n_freq)
        # Width: 1-5 adjacent channels
        _lo, _hi = _scale_range(1, 6, n_freq)
        width = np.random.randint(_lo, _hi)
        # SNR relative to local background noise (log-uniform for balanced weak/strong)
        snr = log_uniform(1.5, 12.0)
        
        freq_start = max(0, freq_center - width // 2)
        freq_end = min(n_freq, freq_center + width // 2 + 1)
        
        # Generate RFI intensity (additive on top of background)
        for f in range(freq_start, freq_end):
            # Intensity varies slightly across width (spectral envelope)
            dist_from_center = abs(f - freq_center) / max(width / 2, 1)
            envelope = np.exp(-0.5 * dist_from_center ** 2)
            intensity = snr * local_sigma[f] * envelope
            
            # Add slight time-domain fluctuation (not perfectly constant)
            time_fluctuation = 1.0 + np.random.normal(0, 0.05, n_time)
            rfi_layer[f, :] += intensity * np.maximum(time_fluctuation, 0)
            mask[f, :] = 1
    
    return rfi_layer, mask


def inject_wideband_burst_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_bursts=None):
    """
    Type 2: Instantaneous Wideband RFI (Vertical Lines / Columns)
    
    Physical source: Radar pulses, lightning strikes, electric spark discharges,
    equipment switching transients, telescope slewing artifacts.
    
    Appearance: Vertical lines spanning many/all frequency channels but lasting
    only 1-3 time bins.
    """
    if n_bursts is None:
        n_bursts = np.random.poisson(lam=1)
    n_bursts = min(n_bursts, 8)
    
    for _ in range(n_bursts):
        # Random time position
        t_center = np.random.randint(0, n_time)
        # Duration: 1-3 time bins
        _lo, _hi = _scale_range(1, 4, n_time)
        t_width = np.random.randint(_lo, _hi)
        # Frequency span: 50%-100% of all channels
        f_span_frac = np.random.uniform(0.5, 1.0)
        f_span = int(f_span_frac * n_freq)
        f_start = np.random.randint(0, max(1, n_freq - f_span))
        f_end = min(n_freq, f_start + f_span)
        # SNR (typically strong)
        snr = log_uniform(3.0, 20.0)
        
        t_start = max(0, t_center - t_width // 2)
        t_end = min(n_time, t_center + t_width // 2 + 1)
        
        for f in range(f_start, f_end):
            intensity = snr * local_sigma[f]
            # Slight frequency-dependent intensity variation
            freq_modulation = 1.0 + np.random.normal(0, 0.1)
            rfi_layer[f, t_start:t_end] += intensity * max(freq_modulation, 0.1)
            mask[f, t_start:t_end] = 1
    
    return rfi_layer, mask


def inject_broadband_block_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_blocks=None):
    """
    Type 3: Broadband RFI (Rectangular Blocks)
    
    Physical source: Satellite passes, Wi-Fi access points, switch-mode power supplies.
    
    Appearance: Rectangular patches covering a range of frequency channels for a
    range of time bins. The intensity within the block has internal structure
    (not perfectly uniform).
    """
    if n_blocks is None:
        n_blocks = np.random.poisson(lam=0.8)
    n_blocks = min(n_blocks, 5)
    
    for _ in range(n_blocks):
        # Frequency extent: 30-300 channels
        _lo, _hi = _scale_range(30, 301, n_freq)
        f_width = np.random.randint(_lo, min(_hi, max(_lo + 1, n_freq // 2)))
        f_start = np.random.randint(0, max(1, n_freq - f_width))
        f_end = min(n_freq, f_start + f_width)
        
        # Time extent: 10%-80% of total time
        t_frac = np.random.uniform(0.1, 0.8)
        t_width = max(5, int(t_frac * n_time))
        t_start = np.random.randint(0, max(1, n_time - t_width))
        t_end = min(n_time, t_start + t_width)
        
        # SNR
        snr = log_uniform(1.5, 8.0)
        
        # Internal texture: not perfectly uniform (real broadband RFI has spectral structure)
        internal_texture = np.random.normal(1.0, 0.2, size=(f_end - f_start, t_end - t_start))
        internal_texture = gaussian_filter(internal_texture, sigma=3.0)
        internal_texture = np.maximum(internal_texture, 0.1)
        
        # Spectral envelope (intensity falls off from center of block)
        f_centers = np.arange(f_end - f_start) - (f_end - f_start) / 2
        spectral_envelope = np.exp(-0.5 * (f_centers / max((f_end - f_start) / 3, 1)) ** 2)
        
        for fi, f in enumerate(range(f_start, f_end)):
            intensity = snr * local_sigma[f] * spectral_envelope[fi]
            rfi_layer[f, t_start:t_end] += intensity * internal_texture[fi, :]
            mask[f, t_start:t_end] = 1
    
    return rfi_layer, mask


def inject_persistent_band_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_bands=None):
    """
    Type 4: Constant Broadband RFI (Persistent Wide Horizontal Bands)
    
    Physical source: Satellite navigation systems (GPS L2 at 1227.6 MHz,
    GLONASS at 1246 MHz, BeiDou at 1268 MHz) always transmitting.
    
    Appearance: Wide horizontal bands (20-100+ channels) persisting across
    the entire time axis. Most prominent RFI features in FAST L-band data.
    """
    if n_bands is None:
        n_bands = np.random.poisson(lam=1.5)
    n_bands = min(n_bands, 6)
    
    for _ in range(n_bands):
        # Width: 10-100 channels
        _lo, _hi = _scale_range(10, 101, n_freq)
        band_width = np.random.randint(_lo, min(_hi, max(_lo + 1, n_freq // 4)))
        # Center position (preferentially around satellite bands but randomized)
        if np.random.random() < 0.5:
            # Satellite band region (~1160-1300 MHz mapped to channel indices)
            freq_frac = np.random.uniform(0.32, 0.60)  # ~1160 to 1300 MHz
            band_center = int(freq_frac * n_freq)
        else:
            band_center = np.random.randint(band_width, n_freq - band_width)
        
        band_start = max(0, band_center - band_width // 2)
        band_end = min(n_freq, band_center + band_width // 2 + 1)
        
        # SNR (typically strong for persistent satellite signals)
        snr = log_uniform(2.0, 15.0)
        
        # Spectral envelope (Gaussian-shaped — intensity falls off from center)
        for f in range(band_start, band_end):
            dist = abs(f - band_center) / max(band_width / 2, 1)
            envelope = np.exp(-0.5 * (dist * 1.5) ** 2)
            intensity = snr * local_sigma[f] * envelope
            
            # Slight temporal fluctuation
            time_var = 1.0 + np.random.normal(0, 0.03, n_time)
            rfi_layer[f, :] += intensity * np.maximum(time_var, 0)
            mask[f, :] = 1
    
    return rfi_layer, mask


def inject_scattered_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_events=None):
    """
    Type 5: Occasional Instantaneous Narrowband RFI (Random Scattered Pixels/Dots)
    
    Physical source: Sporadic electronics interference, short-duration narrowband
    transmissions (taxi radio, satellite sidelobe), cosmic ray hits.
    
    Appearance: Individual pixels or small clusters (2-5 pixels) scattered
    randomly across the image. Isolated — no pattern.
    """
    if n_events is None:
        n_events = np.random.poisson(lam=15)
    n_events = min(n_events, 80)
    
    for _ in range(n_events):
        # Random position
        f_center = np.random.randint(0, n_freq)
        t_center = np.random.randint(0, n_time)
        # Cluster size: 1-5 pixels in each direction
        _lo, _hi = _scale_range(1, 6, n_freq)
        cluster_f = np.random.randint(_lo, _hi)
        _lo, _hi = _scale_range(1, 6, n_time)
        cluster_t = np.random.randint(_lo, _hi)
        # SNR
        snr = log_uniform(2.0, 8.0)
        
        f_start = max(0, f_center - cluster_f // 2)
        f_end = min(n_freq, f_center + cluster_f // 2 + 1)
        t_start = max(0, t_center - cluster_t // 2)
        t_end = min(n_time, t_center + cluster_t // 2 + 1)
        
        for f in range(f_start, f_end):
            intensity = snr * local_sigma[f]
            rfi_layer[f, t_start:t_end] += intensity
            mask[f, t_start:t_end] = 1
    
    return rfi_layer, mask


def inject_blob_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_blobs=None):
    """
    Type 6: Blob / Transient Broadband RFI (Irregular-Shaped Short Patches)
    
    Physical source: Short-duration satellite sidelobe illumination, transient
    equipment interference, intermittent broadband emitters. These appear as
    small, irregular bright patches localized in BOTH frequency and time.
    
    Appearance: Small blobs/patches (5-40 channels × 3-20 time bins) with
    irregular, non-rectangular shapes and variable internal intensity.
    Unlike narrowband (full time span) or wideband bursts (full freq span),
    blobs are confined to a small region in both dimensions.
    """
    if n_blobs is None:
        n_blobs = np.random.poisson(lam=3)
    n_blobs = min(n_blobs, 15)
    
    for _ in range(n_blobs):
        # Blob bounding box size (small patches, not full rows/columns)
        _lo, _hi = _scale_range(5, 40, n_freq)
        blob_f_size = np.random.randint(_lo, _hi)   # scaled from 5-40 ch @1024
        _lo, _hi = _scale_range(3, 20, n_time)
        blob_t_size = np.random.randint(_lo, _hi)    # scaled from 3-20 bins @1024
        
        # Random center position
        f_center = np.random.randint(blob_f_size, n_freq - blob_f_size)
        t_center = np.random.randint(blob_t_size, n_time - blob_t_size)
        
        # SNR (variable — some blobs are faint, some are bright)
        snr = log_uniform(2.0, 12.0)
        
        # Create irregular shape using a random elliptical mask with rotation
        # This ensures blobs are NOT perfect rectangles
        yy, xx = np.mgrid[-blob_f_size:blob_f_size+1, -blob_t_size:blob_t_size+1]
        
        # Random ellipse parameters for irregular shape
        a = np.random.uniform(0.6, 1.0) * blob_f_size  # semi-major axis (freq)
        b = np.random.uniform(0.6, 1.0) * blob_t_size   # semi-minor axis (time)
        theta = np.random.uniform(0, np.pi)               # rotation angle
        
        # Rotated ellipse equation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xx_rot = cos_t * xx + sin_t * yy
        yy_rot = -sin_t * xx + cos_t * yy
        ellipse_dist = (xx_rot / max(b, 1))**2 + (yy_rot / max(a, 1))**2
        
        # Soft-edged elliptical blob (Gaussian falloff from center)
        blob_envelope = np.exp(-0.5 * ellipse_dist)
        # Threshold to create irregular boundary (not perfectly smooth)
        irregularity = np.random.uniform(0.3, 0.6)
        blob_mask_local = blob_envelope > irregularity
        
        # Add internal intensity texture (non-uniform brightness within blob)
        internal_texture = np.random.normal(1.0, 0.25, size=blob_envelope.shape)
        internal_texture = gaussian_filter(internal_texture, sigma=1.5)
        internal_texture = np.maximum(internal_texture, 0.1)
        
        # Map blob onto the full image
        for dy in range(-blob_f_size, blob_f_size + 1):
            for dx in range(-blob_t_size, blob_t_size + 1):
                local_y = dy + blob_f_size
                local_x = dx + blob_t_size
                if local_y < 0 or local_y >= blob_mask_local.shape[0]:
                    continue
                if local_x < 0 or local_x >= blob_mask_local.shape[1]:
                    continue
                if not blob_mask_local[local_y, local_x]:
                    continue
                
                f_idx = f_center + dy
                t_idx = t_center + dx
                if 0 <= f_idx < n_freq and 0 <= t_idx < n_time:
                    intensity = snr * local_sigma[f_idx] * blob_envelope[local_y, local_x] * internal_texture[local_y, local_x]
                    rfi_layer[f_idx, t_idx] += intensity
                    mask[f_idx, t_idx] = 1
    
    return rfi_layer, mask


# ============================================================================
# Section 3: Full Spectrogram Generation Pipeline
# ============================================================================

def generate_spectrogram(n_freq=1024, n_time=1024, rfi_density=None):
    """
    Generates a single synthetic spectrogram with injected RFI and ground truth mask.
    
    Parameters:
        n_freq: Number of frequency channels (rows)
        n_time: Number of time bins (columns)
        rfi_density: One of 'clean', 'light', 'moderate', 'heavy', or None (random)
    
    Returns FOUR values (the docstring previously listed three):
        spectrogram:  2D array (n_freq, n_time) float32 — the contaminated image
        mask:         2D array (n_freq, n_time) uint8 — binary ground truth (1=RFI)
        strength_map: 2D array (n_freq, n_time) float16 — injected RFI amplitude
                      divided by that channel's local noise sigma
        metadata:     dict with generation parameters for reproducibility

    NOTE ON HOW THE LABEL IS DEFINED (this matters for interpreting any score):
    the mask is the hard injected footprint OR'd with `rfi_layer > 0.5*local_sigma`
    (step 4 below). The ground truth is therefore a deterministic, noise-free
    threshold on a smooth latent field, with zero label ambiguity: no
    clean-labelled pixel carries more than 0.5 sigma of injected RFI, by
    construction. Real telescope flags are not like this. See
    `experiments/dataset_difficulty.py` and `AUDIT_REPORT.md`.
    """
    # Determine RFI density if not specified.
    # Thresholds tuned to sit inside the guide's recommended ranges:
    # clean ~5-10%, light ~25-30%, moderate ~35-40%, heavy ~20-25%.
    if rfi_density is None:
        r = np.random.random()
        if r < 0.08:
            rfi_density = 'clean'
        elif r < 0.37:
            rfi_density = 'light'
        elif r < 0.75:
            rfi_density = 'moderate'
        else:
            rfi_density = 'heavy'
    
    # 1. Generate pure thermal noise background
    pure_signal, gain_curve, local_sigma = generate_pure_signal(n_freq, n_time)
    
    # 2. Initialize RFI layer and mask
    rfi_layer = np.zeros((n_freq, n_time), dtype=np.float64)
    mask = np.zeros((n_freq, n_time), dtype=np.uint8)
    
    # 3. Inject RFI based on density category.
    #
    # Each RFI type's *expected count* scales with density tier via a Poisson
    # lambda, and every type is drawn independently for every tier (including
    # zero as a valid outcome). This replaces the previous design where e.g.
    # wideband bursts were only ever injected in 'heavy' images and always
    # forced to be present (1-3, never 0) there. That design meant RFI type
    # was almost perfectly confounded with density tier — a model could learn
    # "many co-occurring types = heavy" as a shortcut instead of learning to
    # recognize each morphology independently, and it also meant a lone
    # wideband burst on an otherwise clean/light background (physically
    # realistic — e.g. a single radar pulse) never appeared in the dataset.
    # With independent Poisson draws, occasional low-lambda occurrences leak
    # across tiers, and even the highest tier can legitimately (if rarely)
    # skip a type — matching how real observations behave.
    density_lambdas = {
        # (narrowband, wideband_burst, broadband_block, persistent_band, blob, scattered)
        'clean':    dict(nb=0.0, wb=0.0,  bb=0.0,  pb=0.0,  bl=0.0, sc=0.0),
        'light':    dict(nb=2.0, wb=0.05, bb=0.15, pb=0.25, bl=0.4, sc=3.0),
        'moderate': dict(nb=6.0, wb=0.3,  bb=1.0,  pb=1.5,  bl=2.0, sc=10.0),
        'heavy':    dict(nb=11.0, wb=1.5, bb=2.2,  pb=3.0,  bl=5.0, sc=25.0),
    }
    lam = density_lambdas[rfi_density]
    
    metadata = {'rfi_density': rfi_density, 'rfi_types': []}
    
    if rfi_density != 'clean':
        n_nb = np.random.poisson(lam=lam['nb']) if lam['nb'] > 0 else 0
        if n_nb > 0:
            rfi_layer, mask = inject_narrowband_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_lines=n_nb)
            metadata['rfi_types'].append(f'narrowband x{n_nb}')
        
        n_wb = np.random.poisson(lam=lam['wb']) if lam['wb'] > 0 else 0
        if n_wb > 0:
            rfi_layer, mask = inject_wideband_burst_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_bursts=n_wb)
            metadata['rfi_types'].append(f'wideband_burst x{n_wb}')
        
        n_bb = np.random.poisson(lam=lam['bb']) if lam['bb'] > 0 else 0
        if n_bb > 0:
            rfi_layer, mask = inject_broadband_block_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_blocks=n_bb)
            metadata['rfi_types'].append(f'broadband_block x{n_bb}')
        
        n_pb = np.random.poisson(lam=lam['pb']) if lam['pb'] > 0 else 0
        if n_pb > 0:
            rfi_layer, mask = inject_persistent_band_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_bands=n_pb)
            metadata['rfi_types'].append(f'persistent_band x{n_pb}')
        
        n_bl = np.random.poisson(lam=lam['bl']) if lam['bl'] > 0 else 0
        if n_bl > 0:
            rfi_layer, mask = inject_blob_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_blobs=n_bl)
            metadata['rfi_types'].append(f'blob x{n_bl}')
        
        n_sc = np.random.poisson(lam=lam['sc']) if lam['sc'] > 0 else 0
        if n_sc > 0:
            rfi_layer, mask = inject_scattered_rfi(rfi_layer, mask, n_freq, n_time, local_sigma, n_events=n_sc)
            metadata['rfi_types'].append(f'scattered x{n_sc}')
    
    # 4. Apply slight Gaussian blur to RFI layer (realistic edge profiles — spectral leakage)
    if np.any(rfi_layer > 0):
        blur_sigma = np.random.uniform(0.3, 0.7)
        rfi_layer = gaussian_filter(rfi_layer, sigma=blur_sigma)
        
        # Re-threshold mask to account for blur spreading.
        # Blurring smears RFI power into neighboring pixels that were not in the
        # original hard-edged mask. If we leave the mask untouched, those pixels
        # end up carrying a real (if small) RFI contribution while being labeled
        # "clean" — a genuine mismatch between pixel values and ground truth.
        # Fix: expand the mask to include any pixel where the blurred RFI power
        # is non-negligible relative to that channel's local noise level.
        leak_threshold = 0.5 * local_sigma[:, np.newaxis]  # 0.5-sigma leak threshold
        mask = np.where((mask == 1) | (rfi_layer > leak_threshold), 1, 0).astype(np.uint8)
    
    # 5. Combine: spectrogram = pure_signal + rfi_layer (ADDITIVE, not replacement)
    spectrogram = pure_signal + rfi_layer
    
    # 5b. Per-pixel RFI STRENGTH map, in units of the local noise sigma.
    #
    # This is essential and was previously missing. `rfi_density`
    # (clean/light/moderate/heavy) records HOW MANY pixels are contaminated,
    # NOT HOW STRONG that contamination is -- they are independent axes. A
    # 'light' image can contain very bright RFI; a 'heavy' image can be full
    # of faint RFI. Without this map it is impossible to report metrics
    # separately for weak / medium / strong RFI, which means the central
    # claim "the model struggles with weak RFI" cannot be tested at all.
    #
    # strength[i,j] = injected RFI amplitude at that pixel / local noise sigma
    # of that frequency channel. So strength=1.0 means the RFI is exactly as
    # large as the typical noise fluctuation there (essentially invisible),
    # while strength=10 means it towers over the noise.
    strength_map = (rfi_layer / local_sigma[:, np.newaxis]).astype(np.float16)
    
    # 6. Compute metadata
    rfi_fraction = np.sum(mask) / mask.size
    metadata['rfi_fraction'] = float(rfi_fraction)
    metadata['n_freq'] = n_freq
    metadata['n_time'] = n_time
    
    # Strength statistics over the RFI-labelled pixels only
    m = mask.astype(bool)
    if m.sum() > 0:
        s = strength_map[m].astype(np.float32)
        metadata['strength_median'] = float(np.median(s))
        metadata['strength_p10'] = float(np.percentile(s, 10))
        metadata['strength_p90'] = float(np.percentile(s, 90))
        # Fractions falling in each strength band (thresholds justified in the
        # dataset audit: measured RFI-pixel strength quartiles are ~2.3 / 3.7 / 6.3 sigma)
        metadata['frac_weak'] = float((s < 3.0).mean())
        metadata['frac_medium'] = float(((s >= 3.0) & (s < 7.0)).mean())
        metadata['frac_strong'] = float((s >= 7.0).mean())
    else:
        for k in ['strength_median', 'strength_p10', 'strength_p90',
                  'frac_weak', 'frac_medium', 'frac_strong']:
            metadata[k] = 0.0
    
    return spectrogram.astype(np.float32), mask, strength_map, metadata


# ============================================================================
# Section 4: Visualization & Preview
# ============================================================================

def plot_preview(spectrogram, mask, metadata, filename):
    """
    Creates a side-by-side preview plot: spectrogram + ground truth mask.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Original spectrogram (with RFI)
    im0 = axes[0].imshow(spectrogram, aspect='auto', cmap='hot', origin='upper')
    axes[0].set_title('Spectrogram (with RFI)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Time Bin')
    axes[0].set_ylabel('Frequency Channel')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Panel 2: Ground truth mask
    im1 = axes[1].imshow(mask, aspect='auto', cmap='Blues', vmin=0, vmax=1, origin='upper')
    axes[1].set_title(f'Ground Truth Mask ({metadata["rfi_fraction"]*100:.1f}% RFI)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time Bin')
    axes[1].set_ylabel('Frequency Channel')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Panel 3: Cleaned spectrogram (RFI masked out in cyan)
    palette = copy(plt.cm.hot)
    palette.set_bad('cyan', 1.0)
    masked_data = np.ma.array(spectrogram, mask=mask)
    vmax = np.percentile(spectrogram[mask == 0], 98) if np.any(mask == 0) else np.max(spectrogram)
    im2 = axes[2].imshow(masked_data, aspect='auto', cmap=palette, vmin=0, vmax=vmax, origin='upper')
    axes[2].set_title(f'Cleaned (RFI masked) — {metadata["rfi_density"]}', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Time Bin')
    axes[2].set_ylabel('Frequency Channel')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Annotation
    rfi_types_str = ', '.join(metadata['rfi_types']) if metadata['rfi_types'] else 'None (clean)'
    fig.suptitle(f'RFI Types: {rfi_types_str}', fontsize=10, y=0.02, color='gray')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Section 5: Dataset Generation Main Loop
# ============================================================================

def generate_dataset(output_dir, n_images=1000, n_freq=1024, n_time=1024, n_previews=20, seed=None):
    """
    Generates the complete synthetic dataset with train/val/test splits.
    
    Parameters:
        output_dir: Root directory for the dataset
        seed: The RNG seed used for this run (recorded in the stats file for reproducibility)
        n_images: Total number of images to generate
        n_freq: Number of frequency channels per image
        n_time: Number of time bins per image
        n_previews: Number of preview PNG images to generate
    """
    # Create directory structure
    splits = {
        'train': int(0.70 * n_images),
        'val': int(0.15 * n_images),
        'test': n_images - int(0.70 * n_images) - int(0.15 * n_images)
    }
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'strength'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'preview'), exist_ok=True)
    
    print(f"=" * 70)
    print(f"Synthetic Radio Astronomy Spectrogram Dataset Generator")
    print(f"=" * 70)
    print(f"  Image size:      {n_freq} x {n_time}")
    print(f"  Total images:    {n_images}")
    print(f"  Train:           {splits['train']}")
    print(f"  Validation:      {splits['val']}")
    print(f"  Test:            {splits['test']}")
    print(f"  Preview images:  {n_previews}")
    print(f"  Output dir:      {output_dir}")
    print(f"=" * 70)
    
    # Statistics tracking
    stats = {
        'clean': 0, 'light': 0, 'moderate': 0, 'heavy': 0,
        'rfi_fractions': [],
        'rfi_type_counts': {}
    }
    
    global_idx = 0
    
    for split_name, split_count in splits.items():
        print(f"\nGenerating {split_name} set ({split_count} images)...")
        
        meta_path = os.path.join(output_dir, split_name, 'metadata.jsonl')
        with open(meta_path, 'w') as meta_f:
            for i in range(split_count):
                # Generate one spectrogram
                spectrogram, mask, strength_map, metadata = generate_spectrogram(n_freq, n_time)
                
                # Save as .npy files
                img_path = os.path.join(output_dir, split_name, 'images', f'spectrogram_{global_idx:04d}.npy')
                mask_path = os.path.join(output_dir, split_name, 'masks', f'mask_{global_idx:04d}.npy')
                strength_path = os.path.join(output_dir, split_name, 'strength', f'strength_{global_idx:04d}.npy')
                np.save(img_path, spectrogram)
                np.save(mask_path, mask)
                np.save(strength_path, strength_map)
                
                # Generate preview for the first n_previews images
                if global_idx < n_previews:
                    preview_path = os.path.join(output_dir, 'preview', f'preview_{global_idx:04d}.png')
                    plot_preview(spectrogram, mask, metadata, preview_path)
                
                # Persist per-image metadata for later, exact benchmark reporting
                # (no re-estimation needed — this is the ground truth of what was
                # actually generated for this specific image)
                record = {
                    'global_idx': global_idx,
                    'split': split_name,
                    'rfi_density': metadata['rfi_density'],
                    'rfi_types': metadata['rfi_types'],
                    'rfi_fraction': metadata['rfi_fraction'],
                    'strength_median': metadata['strength_median'],
                    'strength_p10': metadata['strength_p10'],
                    'strength_p90': metadata['strength_p90'],
                    'frac_weak': metadata['frac_weak'],
                    'frac_medium': metadata['frac_medium'],
                    'frac_strong': metadata['frac_strong'],
                }
                meta_f.write(json.dumps(record) + '\n')
                
                # Track statistics
                stats[metadata['rfi_density']] += 1
                stats['rfi_fractions'].append(metadata['rfi_fraction'])
                for rfi_type in metadata['rfi_types']:
                    type_name = rfi_type.split(' x')[0]
                    stats['rfi_type_counts'][type_name] = stats['rfi_type_counts'].get(type_name, 0) + 1
                
                global_idx += 1
                
                # Progress bar
                if (i + 1) % 50 == 0 or (i + 1) == split_count:
                    print(f"  [{split_name}] {i+1}/{split_count} images generated")
    
    # Print summary statistics
    print(f"\n{'=' * 70}")
    print(f"Dataset Generation Complete!")
    print(f"{'=' * 70}")
    
    print(f"\nRFI Density Distribution:")
    for density in ['clean', 'light', 'moderate', 'heavy']:
        count = stats[density]
        pct = 100 * count / n_images
        print(f"  {density:>10s}: {count:4d} images ({pct:5.1f}%)")
    
    print(f"\nRFI Type Frequency:")
    for rfi_type, count in sorted(stats['rfi_type_counts'].items()):
        print(f"  {rfi_type:>20s}: {count:4d} occurrences")
    
    rfi_fracs = np.array(stats['rfi_fractions'])
    print(f"\nRFI Pixel Fraction Statistics:")
    print(f"  Min:    {100*np.min(rfi_fracs):6.2f}%")
    print(f"  Median: {100*np.median(rfi_fracs):6.2f}%")
    print(f"  Mean:   {100*np.mean(rfi_fracs):6.2f}%")
    print(f"  Max:    {100*np.max(rfi_fracs):6.2f}%")
    
    # Save statistics to a text file
    stats_path = os.path.join(output_dir, 'dataset_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Synthetic Radio Astronomy Spectrogram Dataset\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Image size: {n_freq} x {n_time}\n")
        f.write(f"Total images: {n_images}\n")
        f.write(f"Train: {splits['train']}, Val: {splits['val']}, Test: {splits['test']}\n\n")
        f.write(f"RFI Density Distribution:\n")
        for density in ['clean', 'light', 'moderate', 'heavy']:
            f.write(f"  {density}: {stats[density]} ({100*stats[density]/n_images:.1f}%)\n")
        f.write(f"\nRFI Type Frequency:\n")
        for rfi_type, count in sorted(stats['rfi_type_counts'].items()):
            f.write(f"  {rfi_type}: {count}\n")
        f.write(f"\nRFI Pixel Fraction: min={100*np.min(rfi_fracs):.2f}%, "
                f"median={100*np.median(rfi_fracs):.2f}%, "
                f"mean={100*np.mean(rfi_fracs):.2f}%, "
                f"max={100*np.max(rfi_fracs):.2f}%\n")
    
    print(f"\nStatistics saved to: {stats_path}")
    print(f"Preview images saved to: {os.path.join(output_dir, 'preview')}")
    print(f"\nDone!")
    
    return stats


# ============================================================================
# Section 6: Command Line Interface
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic radio astronomy spectrograms with RFI for ML training'
    )
    parser.add_argument('--n_images', type=int, default=1000,
                        help='Total number of images to generate (default: 1000)')
    parser.add_argument('--size', type=int, default=1024,
                        help='Image size (both freq and time dimensions, default: 1024)')
    parser.add_argument('--n_freq', type=int, default=None,
                        help='Number of frequency channels (overrides --size for freq)')
    parser.add_argument('--n_time', type=int, default=None,
                        help='Number of time bins (overrides --size for time)')
    parser.add_argument('--n_previews', type=int, default=20,
                        help='Number of preview PNG images to generate (default: 20)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory (default: current directory)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')
    parser.add_argument('--preview_only', action='store_true',
                        help='Generate only preview images (20 images, no full dataset)')
    
    args = parser.parse_args()
    
    # Always seed the RNG — either with the user-supplied value, or with a
    # freshly generated one that gets printed and saved so this exact dataset
    # can be regenerated later (e.g. if a professor or reviewer asks for it).
    if args.seed is None:
        args.seed = np.random.randint(0, 2**31 - 1)
        print(f"No --seed given; using auto-generated seed: {args.seed}")
    np.random.seed(args.seed)
    
    # Determine dimensions
    n_freq = args.n_freq if args.n_freq is not None else args.size
    n_time = args.n_time if args.n_time is not None else args.size
    
    # Preview-only mode
    if args.preview_only:
        args.n_images = 20
        args.n_previews = 20
    
    # Generate dataset
    generate_dataset(
        output_dir=args.output_dir,
        n_images=args.n_images,
        n_freq=n_freq,
        n_time=n_time,
        n_previews=args.n_previews,
        seed=args.seed
    )
