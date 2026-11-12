"""Per-peak features from a single amplitude, power, dB, or complex CIR.

The returned DataFrame has one row per candidate. Positions and widths use the
units of ``axis`` (sample bins when omitted). No truth or range calibration is
used. Complex features require the full complex CIR and FrequencyContext;
unavailable features are NaN, never fabricated phase/noise measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths


@dataclass(frozen=True)
class PeakFeatureConfig:
    # Height/prominence thresholds are fractions of the maximum in search_range.
    min_relative_height: float = 0.02
    min_relative_prominence: float = 0.01
    min_peak_distance: float | None = None
    max_peaks: int | None = None
    search_range: tuple[float, float] | None = None
    include_boundary_peaks: bool = True
    refine_positions: bool = True
    scales: tuple[float, ...] = (1.0, 3.0, 5.0)
    core_radius: float = 0.25
    background_radius: float = 10.0
    history_match_radius: float = 1.0
    max_history: int = 5
    max_model_peaks: int = 12
    stability_match_radius: float = 1.0
    perturbations: int = 8
    dropout_fraction: float = 0.10
    random_seed: int = 20261112


@dataclass(frozen=True)
class FrequencyContext:
    """How the *full* complex CIR was constructed by an N-point IFFT.

    active_bins: measured frequency indices in NumPy FFT order, not CIR indices.
      For 80 tones followed by zero padding to 512: np.arange(80).
    fft_shifted: whether fftshift was applied to the IFFT output.
    weights: positive real weights already applied to these tones, same order as
      active_bins; None means rectangular. Tones with zero weight are omitted.
      The order of active_bins must follow increasing physical frequency; signed
      FFT-bin frequencies must differ by a constant from physical frequencies.

    A cropped CIR, magnitude CIR, or nonlinear log CIR cannot be used here.
    """
    active_bins: Sequence[int]
    fft_shifted: bool = False
    weights: Sequence[float] | None = None


_BASIC = {
    "peak_index": "Original CIR sample index, zero based",
    "position_bin": "Coordinate of the detected sample",
    "position": "Coordinate after optional three-point parabolic refinement",
    "amplitude": "Amplitude at the detected sample; dB input is max-normalized",
    "relative_height": "Amplitude / maximum amplitude within search_range",
    "is_local_maximum": "Candidate is a detected local maximum",
    "is_boundary_peak": "Candidate is at an end of the full CIR",
    "time_rank": "Candidate rank by increasing position, zero based",
    "height_rank": "Candidate rank by decreasing amplitude, zero based",
    "prominence_ratio": "Topographic amplitude prominence / ROI maximum",
    "half_prominence_width": "Width at half amplitude prominence, axis units",
    "left_half_prominence_width": "Left extent of half-prominence width",
    "right_half_prominence_width": "Right extent of half-prominence width",
    "half_power_width": "Width at amplitude peak/sqrt(2), not half prominence",
    "half_power_asymmetry": "(right width-left width)/(right width+left width)",
    "left_40pct_position": "Left crossing of 40% of this candidate's amplitude",
    "rise_10_to_90": "Left-side 10%-to-90% amplitude rise distance",
    "normalized_curvature": "Negative second amplitude difference / ROI max / dx^2",
    "left_slope": "Left first amplitude difference / ROI max / dx",
    "right_slope": "Right first amplitude difference / ROI max / dx",
    "candidate_to_local_background": "Amplitude / local median; not RF SNR",
    "local_background_ratio": "Local median amplitude / ROI maximum",
    "local_background_robust_contrast": "(amplitude-median)/(1.4826 MAD); not a p-value",
    "background_sample_count": "Samples in local background annulus",
    "above_background_run_length": "Contiguous span around candidate above local median+3*1.4826*MAD; heuristic",
    "above_background_run_truncated": "Above-background span touches an ROI boundary",
    "gap_from_strongest": "Strongest sample position minus candidate position",
    "height_to_strongest": "Candidate amplitude / strongest amplitude",
    "left_neighbor_gap": "Distance to nearest earlier local maximum before thresholds",
    "right_neighbor_gap": "Distance to nearest later local maximum before thresholds",
    "nearest_neighbor_gap": "Minimum of available left/right neighbor distances",
    "nearest_neighbor_height_ratio": "Nearest other maximum amplitude / candidate amplitude",
    "earlier_peak_count": "Earlier local maxima count before height/prominence filters",
    "earlier_energy_fraction": "ROI power before candidate minus core_radius / total power",
}
_GLOBAL = {
    "global_power_entropy": "Normalized discrete entropy of ROI power distribution",
    "global_mean_position": "Power-weighted mean coordinate in ROI",
    "global_rms_delay_spread": "Power-weighted coordinate standard deviation",
    "global_energy_span_90": "Difference between 95% and 5% energy coordinates",
    "global_amplitude_skewness": "Centered amplitude skewness in ROI",
    "global_amplitude_kurtosis": "Centered Pearson amplitude kurtosis (not excess)",
    "global_background_ratio": "ROI median amplitude / maximum amplitude",
    "global_peak_count": "Local maximum count in ROI before candidate filtering",
    "global_strongest_position": "Coordinate of ROI maximum, optionally refined",
}
_MODEL = {
    "single_path_template_nmse": "Local complex template fit NMSE, or envelope-shape NMSE",
    "other_path_fit_nmse": "Raw-tone NMSE fitting reference candidates except this one",
    "joint_fit_nmse": "Raw-tone NMSE after also fitting this candidate",
    "fit_gain_fraction": "Conditional RSS reduction / RSS before adding candidate",
    "fit_gain_total_energy_fraction": "Conditional RSS reduction / observed tone energy",
    "fit_bic_gain_heuristic": "2M log(RSS_without/RSS_with)-3 log(2M); heuristic only",
    "other_path_explained_amplitude_ratio": "Other fitted paths' amplitude here / observed amplitude",
    "residual_amplitude_ratio": "Residual complex amplitude here / observed amplitude",
    "conditional_residual_coherence": "Tone coherence of residual after fitting other paths",
    "model_reference_count": "Number of other candidate delays fitted jointly",
    "model_dictionary_condition": "Condition number after adding this candidate",
}
_FREQUENCY = {
    "frequency_phase_coherence": "All-tone coherent sum at candidate; related to IFFT height",
    "subband_coherence_mean": "Mean within-band coherence over up to four bands",
    "subband_coherence_min": "Minimum within-band coherence",
    "subband_coherence_std": "Standard deviation of within-band coherence",
    "subband_peak_position_mad": "Median band-maximum offset in local search neighborhood",
    "tone_amplitude_cv": "Raw-tone amplitude std/mean",
    "tone_deep_fade_fraction": "Tone amplitude below 10% of median tone amplitude",
    "spectral_out_of_band_power_fraction": "FFT power outside declared active bins / all FFT power",
    "window_support_fraction": "Matched-peak fraction across rectangular/Hann/Hamming views",
    "window_position_mad": "Median absolute matched-peak offset across window views",
    "window_relative_height_cv": "Matched relative-height std/mean across window views",
    "dropout_support_fraction": "Matched-peak fraction across frequency-dropout views",
    "dropout_position_mad": "Median absolute matched-peak offset across dropout views",
    "dropout_relative_height_cv": "Matched relative-height std/mean across dropout views",
}
_HISTORY = {
    "history_frame_count": "Number of supplied past frames considered",
    "history_support_count": "Past frames with a candidate within matching radius",
    "history_support_fraction": "Support count / supplied past frame count",
    "history_position_mad": "Median absolute position offset, after caller-supplied shifts",
    "history_height_ratio": "Current amplitude / median matched past amplitude",
}
_SCALE = {
    "local_energy_fraction": "Power within candidate +/- radius / ROI power",
    "pre_energy_fraction": "Power in [-radius,-core_radius) / ROI power",
    "post_energy_fraction": "Power in (core_radius,radius] / ROI power",
    "rise_asymmetry": "(post power-pre power)/(post power+pre power)",
    "local_peak_count": "Other local maxima within radius, before filtering",
    "pre_post_mean_contrast": "Post minus pre amplitude mean / ROI max",
    "pre_post_variance_contrast": "Post minus pre amplitude variance / ROI max^2",
    "pre_post_power_entropy_contrast": "Post minus pre normalized discrete power entropy; not the 2015 entropy detector",
    "window_coverage_fraction": "Physical window extent inside the ROI / full window extent",
}
EPS = 1e-12


def _scale_tag(radius: float) -> str:
    return format(float(radius), ".12g").replace(".", "p").replace("-", "m")


def feature_schema(config: PeakFeatureConfig | None = None) -> pd.DataFrame:
    """Column descriptions. Dynamic external-algorithm columns are in df.attrs."""
    cfg = config or PeakFeatureConfig()
    rows = []
    for group, fields, needs in [
        ("peak", _BASIC, "CIR"), ("global", _GLOBAL, "CIR"),
        ("model", _MODEL, "PSF; conditional features also need complex CIR + FrequencyContext"),
        ("frequency", _FREQUENCY, "full complex CIR + FrequencyContext"),
        ("history", _HISTORY, "previous peak DataFrames from the same antenna path"),
    ]:
        rows.extend(dict(feature=k, group=group, requires=needs, description=v)
                    for k, v in fields.items())
    for radius in cfg.scales:
        rows.extend(dict(feature=f"{k}_r{_scale_tag(radius)}", group="multiscale",
                         requires="CIR", description=f"{v}; radius={radius:g} axis units")
                    for k, v in _SCALE.items())
    return pd.DataFrame(rows)


def _validate_config(cfg: PeakFeatureConfig) -> None:
    for name in ["min_relative_height", "min_relative_prominence"]:
        v = getattr(cfg, name)
        if not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not np.isfinite(cfg.dropout_fraction) or not 0 <= cfg.dropout_fraction < 1:
        raise ValueError("dropout_fraction must be finite and in [0, 1)")
    for name in ["core_radius", "background_radius", "history_match_radius", "stability_match_radius"]:
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if cfg.background_radius <= cfg.core_radius:
        raise ValueError("background_radius must exceed core_radius")
    if not cfg.scales or any(not np.isfinite(v) or v <= cfg.core_radius for v in cfg.scales):
        raise ValueError("Every scale must be finite and exceed core_radius")
    if len({_scale_tag(v) for v in cfg.scales}) != len(cfg.scales):
        raise ValueError("scales must be distinct")
    for name in ["max_history", "max_model_peaks", "perturbations"]:
        v = getattr(cfg, name)
        if not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.max_peaks is not None and (not isinstance(cfg.max_peaks, (int, np.integer)) or cfg.max_peaks < 1):
        raise ValueError("max_peaks must be a positive integer or None")
    if cfg.min_peak_distance is not None and (not np.isfinite(cfg.min_peak_distance) or cfg.min_peak_distance <= 0):
        raise ValueError("min_peak_distance must be positive and finite")


def _prepare(cir, axis, input_kind):
    values = np.asarray(cir)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("cir must be a finite 1-D array of at least three samples")
    n = len(values)
    x = np.arange(n, dtype=float) if axis is None else np.asarray(axis, dtype=float)
    if x.shape != (n,) or not np.isfinite(x).all():
        raise ValueError("axis must be finite and have the same length as cir")
    dx = np.diff(x)
    if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-6, atol=abs(dx[0])*1e-8):
        raise ValueError("axis must be strictly increasing and uniformly spaced")
    kind = ("complex" if np.iscomplexobj(values) else "amplitude") if input_kind == "auto" else input_kind
    if kind not in {"complex", "amplitude", "power", "power_db", "amplitude_db"}:
        raise ValueError("Unknown input_kind")
    if np.iscomplexobj(values) and kind != "complex":
        raise ValueError("Complex-valued input requires input_kind='complex' or 'auto'")
    h = np.asarray(values, dtype=np.complex128) if kind == "complex" else None
    db_reference = None
    if kind == "complex":
        amplitude = np.abs(h)
    else:
        real = np.asarray(values, dtype=float)
        if kind.endswith("_db"):
            # Both amplitude dB and power dB correspond to amplitude=10**(dB/20).
            db_reference = float(real.max())
            amplitude = 10.0**((real-db_reference)/20.0)
        else:
            if np.any(real < 0):
                raise ValueError("Linear amplitude/power must be nonnegative; use an explicit dB kind")
            amplitude = np.sqrt(real) if kind == "power" else real.copy()
    scale = float(amplitude.max())
    if scale > 0:
        amplitude = amplitude/scale
        if h is not None:
            h = h/scale
    return amplitude, h, x, float(dx[0]), scale, kind, db_reference


def _local_maxima(a: np.ndarray, boundaries: bool) -> np.ndarray:
    found = find_peaks(a)[0]
    if boundaries:
        ends = [i for i, j in [(0, 1), (len(a)-1, len(a)-2)] if a[i] > a[j] and a[i] > 0]
        found = np.unique(np.r_[found, ends]).astype(int)
    return found


def _refine(a, peak, enabled):
    if enabled and 0 < peak < len(a)-1 and a[peak] >= max(a[peak-1], a[peak+1]):
        denominator = a[peak-1] - 2*a[peak] + a[peak+1]
        if denominator < -EPS:
            return peak + float(np.clip(.5*(a[peak-1]-a[peak+1])/denominator, -.5, .5))
    return float(peak)


def _crossing(a, x, peak, fraction, side):
    level = a[peak]*fraction
    j = peak
    while 0 <= j+side < len(a):
        k = j+side
        if a[k] <= level:
            denominator = a[k]-a[j]
            return float(x[j]+(level-a[j])*(x[k]-x[j])/denominator) if denominator else float(x[k])
        j = k
    return np.nan


def _finite_min(values):
    good = [v for v in values if np.isfinite(v)]
    return min(good) if good else np.nan


def _power_entropy(values):
    if len(values) < 2 or values.sum() <= EPS:
        return np.nan
    p = values/values.sum()
    return float(-np.sum(p*np.log(np.maximum(p, EPS)))/np.log(len(p)))


def _spectrum_data(h, context):
    n = len(h)
    raw_bins = np.asarray(context.active_bins)
    if raw_bins.ndim != 1 or len(raw_bins) < 4 or not np.issubdtype(raw_bins.dtype, np.integer):
        raise ValueError("active_bins must contain at least four integer FFT-bin indices")
    bins = raw_bins.astype(int)
    if len(np.unique(bins)) != len(bins) or np.any((bins < 0) | (bins >= n)):
        raise ValueError("active_bins must be unique indices in [0, N)")
    weights = np.ones(len(bins)) if context.weights is None else np.asarray(context.weights, float)
    if weights.shape != bins.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("weights must be positive, finite and match active_bins")
    k = np.fft.fftfreq(n)[bins]*n
    if np.any(np.diff(k) <= 0):
        raise ValueError("active_bins must follow increasing signed FFT frequency")
    spectrum = np.fft.fft(np.fft.ifftshift(h) if context.fft_shifted else h)
    z = spectrum[bins]/weights
    mask = np.zeros(n, bool); mask[bins] = True
    outside = np.sum(abs(spectrum[~mask])**2)/max(np.sum(abs(spectrum)**2), EPS)
    psf_spectrum = np.zeros(n, complex); psf_spectrum[bins] = weights
    return dict(n=n, bins=bins, k=k, weights=weights, z=z,
                outside=float(outside), psf=np.fft.ifft(psf_spectrum))


def _evaluate(z, weights, k, positions, n):
    return np.exp(2j*np.pi*np.outer(positions, k)/n) @ (weights*z)/n


def _stability_view(a, x, cfg):
    peaks = _local_maxima(a, cfg.include_boundary_peaks)
    roi = np.ones(len(a), bool) if cfg.search_range is None else ((x >= cfg.search_range[0]) & (x <= cfg.search_range[1]))
    maximum = max(float(a[roi].max()), EPS)
    peaks = peaks[roi[peaks] & (a[peaks] >= cfg.min_relative_height*maximum)]
    interior = peaks[(peaks > 0) & (peaks < len(a)-1)]
    if len(interior):
        prom = peak_prominences(a, interior)[0]
        keep = dict(zip(interior, prom >= cfg.min_relative_prominence*maximum))
        peaks = np.asarray([p for p in peaks if p not in keep or keep[p]], int)
    positions = np.asarray([np.interp(_refine(a, p, cfg.refine_positions), np.arange(len(a)), x) for p in peaks])
    return positions, a[peaks]/maximum


def _stability_summary(position, views, radius):
    offsets, heights = [], []
    for positions, relative in views:
        if len(positions):
            j = int(np.argmin(abs(positions-position)))
            if abs(positions[j]-position) <= radius:
                offsets.append(abs(float(positions[j]-position))); heights.append(float(relative[j]))
    return (len(offsets)/len(views), float(np.median(offsets)) if offsets else np.nan,
            float(np.std(heights)/max(np.mean(heights), EPS)) if heights else np.nan)


def _frequency_views(data, x, context, cfg):
    m, n = len(data["z"]), data["n"]
    def view(weights):
        full = np.zeros(n, complex); full[data["bins"]] = data["z"]*weights
        h = np.fft.ifft(full)
        if context.fft_shifted:
            h = np.fft.fftshift(h)
        return _stability_view(abs(h), x, cfg)
    # These windows are applied to recovered raw tones, not multiplied twice.
    windows = [view(w) for w in [np.ones(m), np.hanning(m), np.hamming(m)]]
    rng = np.random.default_rng(cfg.random_seed)
    drops = []
    drop_count = min(m-4, max(1, round(m*cfg.dropout_fraction))) if cfg.dropout_fraction > 0 else 0
    for _ in range(cfg.perturbations):
        w = data["weights"].copy()
        if drop_count:
            w[rng.choice(m, drop_count, replace=False)] = 0
        drops.append(view(w))
    return windows, drops


def _template_nmse(a, h, psf, sample_position, radius_samples):
    n = len(a)
    ids = np.arange(max(0, int(np.floor(sample_position-radius_samples))),
                    min(n, int(np.ceil(sample_position+radius_samples))+1))
    k = np.fft.fftfreq(n)*n
    shifted = np.fft.ifft(np.fft.fft(psf)*np.exp(-2j*np.pi*k*sample_position/n))
    template = shifted[ids] if h is not None else abs(shifted[ids])
    target = h[ids] if h is not None else a[ids]
    denominator = float(np.vdot(template, template).real)
    if denominator <= EPS:
        return np.nan
    coefficient = np.vdot(template, target)/denominator
    return float(np.sum(abs(target-coefficient*template)**2)/max(np.sum(abs(target)**2), EPS))


def _model_features(data, sample_positions, j, strengths, context, cfg):
    # Reference set is magnitude-selected, truth-independent, and bounded in size.
    ordered = np.argsort(-strengths, kind="stable")
    # Leave observation degrees of freedom when very few tones are supplied.
    reference_limit = min(cfg.max_model_peaks, max(0, len(data["z"])//2-1))
    others = [p for p in ordered if p != j][:reference_limit]
    p = sample_positions - (data["n"]//2 if context.fft_shifted else 0)
    z, k, n = data["z"], data["k"], data["n"]
    basis0 = np.exp(-2j*np.pi*np.outer(k, p[others])/n)
    basis1 = np.column_stack([basis0, np.exp(-2j*np.pi*k*p[j]/n)])
    if others:
        coefficients = np.linalg.lstsq(basis0, z, rcond=1e-8)[0]
        fitted0 = basis0 @ coefficients
    else:
        fitted0 = np.zeros_like(z)
    fitted1 = basis1 @ np.linalg.lstsq(basis1, z, rcond=1e-8)[0]
    residual = z-fitted0
    energy = max(float(np.vdot(z, z).real), EPS)
    rss0 = float(np.vdot(residual, residual).real)
    rss1 = float(np.sum(abs(z-fitted1)**2))
    gain = max(0., rss0-rss1)
    # Relative floor prevents infinite heuristic scores for noiseless examples.
    floor = max(EPS*energy, np.finfo(float).tiny)
    actual = _evaluate(z, data["weights"], k, [p[j]], n)[0]
    fitted_here = _evaluate(fitted0, data["weights"], k, [p[j]], n)[0]
    residual_here = actual-fitted_here
    condition = float(np.linalg.cond(basis1))
    return dict(other_path_fit_nmse=rss0/energy, joint_fit_nmse=rss1/energy,
                fit_gain_fraction=gain/max(rss0, floor), fit_gain_total_energy_fraction=gain/energy,
                fit_bic_gain_heuristic=2*len(z)*np.log(max(rss0, floor)/max(rss1, floor))-3*np.log(2*len(z)),
                other_path_explained_amplitude_ratio=abs(fitted_here)/max(abs(actual), EPS),
                residual_amplitude_ratio=abs(residual_here)/max(abs(actual), EPS),
                conditional_residual_coherence=abs(np.sum(residual*np.exp(2j*np.pi*k*p[j]/n)))/max(np.sum(abs(residual)), EPS),
                model_reference_count=len(others), model_dictionary_condition=condition if np.isfinite(condition) else np.nan)


def extract_peak_features(
    cir, axis=None, *, input_kind: str = "auto", config: PeakFeatureConfig | None = None,
    peak_indices: Sequence[int] | None = None,
    frequency_context: FrequencyContext | None = None, psf=None,
    history: Sequence[pd.DataFrame] | None = None,
    history_shifts: Sequence[float] | None = None,
    reference_peak_sets: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Extract one feature row per detected (or explicitly supplied) candidate.

    axis: uniform increasing distances/times, or omitted for sample-bin units.
    input_kind: auto, complex, amplitude, power, power_db, amplitude_db.
    peak_indices: bypass candidate thresholds and spacing; all supplied indices
      are evaluated. Non-maxima have NaN prominence/width. Search range still
      applies and out-of-range candidates are rejected.
    psf: optional length-N system impulse template, centered at index 0 in
      unshifted order. Allows an envelope-template fit for magnitude-only input.
    history: oldest-to-newest prior result DataFrames from the SAME antenna path.
      Only the last max_history frames are read. No history is maintained globally.
    history_shifts: offsets added to those historical positions for motion
      compensation, same length/order as history. No future data are used.
    reference_peak_sets: optional other algorithms' peak coordinates, same axis.

    No reliability labels or probability scores are produced; this is a feature
    extractor. See df.attrs['feature_availability'] and feature_schema().
    """
    cfg = config or PeakFeatureConfig(); _validate_config(cfg)
    a, h, x, dx, scale, kind, db_reference = _prepare(cir, axis, input_kind)
    n = len(a)
    roi = np.ones(n, bool)
    if cfg.search_range is not None:
        bounds = np.asarray(cfg.search_range, float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("search_range must be a finite increasing pair")
        roi = (x >= bounds[0]) & (x <= bounds[1])
        if not roi.any():
            raise ValueError("search_range contains no CIR samples")
    maximum = max(float(a[roi].max()), EPS)
    b = a/maximum
    power = b*b
    total = max(float(power[roi].sum()), EPS)
    all_peaks = _local_maxima(a, cfg.include_boundary_peaks)
    all_peaks = all_peaks[roi[all_peaks]]
    interior = all_peaks[(all_peaks > 0) & (all_peaks < n-1)]
    shape = {}
    if len(interior):
        prominence, lb, rb = peak_prominences(a, interior)
        widths, _, left, right = peak_widths(a, interior, rel_height=.5,
                                            prominence_data=(prominence, lb, rb))
        shape = {int(p): (float(prominence[j]/maximum), float(widths[j]*dx),
                         float((p-left[j])*dx), float((right[j]-p)*dx))
                 for j, p in enumerate(interior)}
    if peak_indices is None:
        selected = [int(p) for p in all_peaks if a[p] > 0 and a[p] >= cfg.min_relative_height*maximum
                    and (p not in shape or shape[p][0] >= cfg.min_relative_prominence)]
        ordered = sorted(selected, key=lambda p: (-a[p], p))
        if cfg.min_peak_distance is not None:
            accepted = []
            for p in ordered:
                if all(abs(x[p]-x[q]) >= cfg.min_peak_distance for q in accepted):
                    accepted.append(p)
            ordered = accepted
        if cfg.max_peaks is not None:
            ordered = ordered[:cfg.max_peaks]
        peaks = np.asarray(sorted(ordered), int)
    else:
        raw = np.asarray(peak_indices)
        if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
            raise ValueError("peak_indices must be a 1-D integer sequence")
        peaks = np.unique(raw.astype(int))
        if np.any((peaks < 0) | (peaks >= n)) or not roi[peaks].all():
            raise ValueError("peak_indices contains indices outside the CIR/search_range")

    data = None
    if frequency_context is not None:
        if h is None:
            raise ValueError("FrequencyContext requires a full complex CIR")
        data = _spectrum_data(h, frequency_context)
    template = None if psf is None else np.asarray(psf, dtype=complex)
    if template is not None and (template.shape != (n,) or not np.isfinite(template).all() or not np.any(abs(template) > 0)):
        raise ValueError("psf must be a finite, nonzero length-N template centered at index 0")
    if template is None and data is not None:
        template = data["psf"]
    if template is not None:
        template = template/abs(template).max()

    past = list(history) if history is not None else []
    shifts = np.zeros(len(past)) if history_shifts is None else np.asarray(history_shifts, float)
    if shifts.shape != (len(past),) or not np.isfinite(shifts).all():
        raise ValueError("history_shifts must be finite and match history length")
    past = list(zip(past, shifts))[-cfg.max_history:]
    for frame, _ in past:
        if not {"position", "amplitude"}.issubset(frame.columns):
            raise ValueError("Each history frame needs position and amplitude columns")
        if not np.isfinite(frame[["position", "amplitude"]].to_numpy(float)).all() or (frame.amplitude < 0).any():
            raise ValueError("Historical positions/amplitudes must be finite; amplitudes nonnegative")

    refs = {}
    for name, positions in (reference_peak_sets or {}).items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Reference method names must be valid identifiers")
        ref = np.asarray(positions, float)
        if ref.ndim != 1 or not np.isfinite(ref).all():
            raise ValueError("Reference peak coordinates must be finite 1-D sequences")
        refs[name] = ref
    columns = list(feature_schema(cfg).feature)
    for name in refs:
        columns += [f"reference_{name}_gap", f"reference_{name}_support"]

    samples = np.asarray([_refine(a, p, cfg.refine_positions) for p in peaks])
    positions = np.interp(samples, np.arange(n), x)
    main = int(np.flatnonzero(roi)[np.argmax(a[roi])])
    main_pos = float(np.interp(_refine(a, main, cfg.refine_positions), np.arange(n), x))
    height_order = np.argsort(-a[peaks], kind="stable")
    height_ranks = np.empty(len(peaks), int); height_ranks[height_order] = np.arange(len(peaks))
    prob = power[roi]/total
    entropy = -np.sum(prob*np.log(np.maximum(prob, EPS)))/np.log(max(2, roi.sum()))
    mean = float(np.sum(prob*x[roi]))
    spread = float(np.sqrt(np.sum(prob*(x[roi]-mean)**2)))
    cdf = np.cumsum(prob)
    quant = np.interp([.05, .95], cdf, x[roi]) if cdf[-1] > 0 else [np.nan, np.nan]
    centered = b[roi]-b[roi].mean(); variance = float(np.mean(centered**2))
    glob = dict(global_power_entropy=float(entropy), global_mean_position=mean,
                global_rms_delay_spread=spread, global_energy_span_90=float(quant[1]-quant[0]),
                global_amplitude_skewness=float(np.mean(centered**3)/variance**1.5) if variance > EPS else np.nan,
                global_amplitude_kurtosis=float(np.mean(centered**4)/variance**2) if variance > EPS else np.nan,
                global_background_ratio=float(np.median(b[roi])), global_peak_count=len(all_peaks),
                global_strongest_position=main_pos)
    views = _frequency_views(data, x, frequency_context, cfg) if data is not None and len(peaks) else None
    rows = []
    for j, peak in enumerate(peaks):
        pos = positions[j]; delta = x-pos
        row = dict.fromkeys(columns, np.nan)
        row.update(glob)
        row.update(peak_index=int(peak), position_bin=float(x[peak]), position=float(pos),
                   amplitude=float(a[peak]*scale), relative_height=float(b[peak]),
                   is_local_maximum=bool(peak in all_peaks), is_boundary_peak=bool(peak in (0, n-1)),
                   time_rank=j, height_rank=int(height_ranks[j]), gap_from_strongest=main_pos-pos,
                   height_to_strongest=float(b[peak]),
                   earlier_peak_count=int(np.sum(x[all_peaks] < pos-dx*.5)),
                   earlier_energy_fraction=float(power[roi & (delta < -cfg.core_radius)].sum()/total))
        if peak in shape:
            row.update(zip(["prominence_ratio", "half_prominence_width", "left_half_prominence_width",
                            "right_half_prominence_width"], shape[peak]))
        l = _crossing(a, x, peak, 1/np.sqrt(2), -1)
        r = _crossing(a, x, peak, 1/np.sqrt(2), 1)
        width = r-l
        row.update(half_power_width=float(width),
                   half_power_asymmetry=float((r+l-2*x[peak])/width) if np.isfinite(width) and width > 0 else np.nan,
                   left_40pct_position=_crossing(a, x, peak, .4, -1),
                   rise_10_to_90=_crossing(a, x, peak, .9, -1)-_crossing(a, x, peak, .1, -1))
        if 0 < peak < n-1:
            row.update(normalized_curvature=float((2*b[peak]-b[peak-1]-b[peak+1])/dx**2),
                       left_slope=float((b[peak]-b[peak-1])/dx), right_slope=float((b[peak+1]-b[peak])/dx))
        local_bg = b[roi & (abs(delta) > cfg.core_radius) & (abs(delta) <= cfg.background_radius)]
        row["background_sample_count"] = len(local_bg)
        if len(local_bg):
            median = float(np.median(local_bg)); mad = float(np.median(abs(local_bg-median)))
            row.update(candidate_to_local_background=float(b[peak]/max(median, EPS)),
                       local_background_ratio=median,
                       local_background_robust_contrast=float((b[peak]-median)/max(1.4826*mad, EPS)))
            threshold = median+3*1.4826*mad
            lo = hi = int(peak)
            if b[peak] > threshold:
                while lo > 0 and roi[lo-1] and b[lo-1] > threshold:
                    lo -= 1
                while hi+1 < n and roi[hi+1] and b[hi+1] > threshold:
                    hi += 1
                row["above_background_run_length"] = float(x[hi]-x[lo]+dx)
                row["above_background_run_truncated"] = bool(lo == 0 or hi == n-1 or not roi[max(0, lo-1)] or not roi[min(n-1, hi+1)])
            else:
                row["above_background_run_length"] = 0.
                row["above_background_run_truncated"] = False
        other = all_peaks[all_peaks != peak]
        left = x[other][x[other] < pos]; right = x[other][x[other] > pos]
        row["left_neighbor_gap"] = float(pos-left.max()) if len(left) else np.nan
        row["right_neighbor_gap"] = float(right.min()-pos) if len(right) else np.nan
        row["nearest_neighbor_gap"] = _finite_min([row["left_neighbor_gap"], row["right_neighbor_gap"]])
        if len(other):
            nearest = other[np.argmin(abs(x[other]-pos))]
            row["nearest_neighbor_height_ratio"] = float(a[nearest]/max(a[peak], EPS))
        for radius in cfg.scales:
            pre = power[roi & (delta >= -radius) & (delta < -cfg.core_radius)]
            post = power[roi & (delta > cfg.core_radius) & (delta <= radius)]
            # Means and variances here describe amplitude, not RF noise.
            pre_a = b[roi & (delta >= -radius) & (delta < -cfg.core_radius)]
            post_a = b[roi & (delta > cfg.core_radius) & (delta <= radius)]
            values = dict(local_energy_fraction=float(power[roi & (abs(delta) <= radius)].sum()/total),
                          pre_energy_fraction=float(pre.sum()/total), post_energy_fraction=float(post.sum()/total),
                          rise_asymmetry=float((post.sum()-pre.sum())/max(post.sum()+pre.sum(), EPS)),
                          local_peak_count=int(np.sum(abs(x[other]-pos) <= radius)),
                          pre_post_mean_contrast=float(post_a.mean()-pre_a.mean()) if len(pre_a) and len(post_a) else np.nan,
                          pre_post_variance_contrast=float(post_a.var()-pre_a.var()) if len(pre_a) and len(post_a) else np.nan,
                          pre_post_power_entropy_contrast=_power_entropy(post)-_power_entropy(pre),
                          window_coverage_fraction=float(max(0., min(x[roi][-1], pos+radius)-max(x[roi][0], pos-radius))/(2*radius)))
            row.update({f"{k}_r{_scale_tag(radius)}": v for k, v in values.items()})
        if template is not None:
            row["single_path_template_nmse"] = _template_nmse(a, h, template, samples[j], max(cfg.scales)/dx)
        if data is not None:
            row.update(_model_features(data, samples, j, a[peaks], frequency_context, cfg))
            p = samples[j]-(n//2 if frequency_context.fft_shifted else 0)
            rotated = data["z"]*np.exp(2j*np.pi*data["k"]*p/n)
            coherence = abs(rotated.sum())/max(abs(rotated).sum(), EPS)
            bands = np.array_split(np.arange(len(rotated)), min(4, len(rotated)//2))
            band_coh, band_offset = [], []
            neighborhood = np.flatnonzero(roi & (abs(delta) <= cfg.stability_match_radius))
            if not len(neighborhood):
                neighborhood = np.asarray([peak])
            local_p = neighborhood-(n//2 if frequency_context.fft_shifted else 0)
            for ids in bands:
                band_coh.append(abs(rotated[ids].sum())/max(abs(rotated[ids]).sum(), EPS))
                local = abs(_evaluate(data["z"][ids], np.ones(len(ids)), data["k"][ids], local_p, n))
                band_offset.append(abs(float(x[neighborhood[np.argmax(local)]]-pos)))
            tone_amp = abs(data["z"])
            row.update(frequency_phase_coherence=float(coherence), subband_coherence_mean=float(np.mean(band_coh)),
                       subband_coherence_min=float(np.min(band_coh)), subband_coherence_std=float(np.std(band_coh)),
                       subband_peak_position_mad=float(np.median(band_offset)),
                       tone_amplitude_cv=float(tone_amp.std()/max(tone_amp.mean(), EPS)),
                       tone_deep_fade_fraction=float(np.mean(tone_amp < .1*np.median(tone_amp))),
                       spectral_out_of_band_power_fraction=data["outside"])
            for prefix, family in zip(["window", "dropout"], views):
                support, mad, cv = _stability_summary(pos, family, cfg.stability_match_radius)
                row.update({prefix+"_support_fraction": support, prefix+"_position_mad": mad,
                            prefix+"_relative_height_cv": cv})
        if history is not None:
            matches = []
            for frame, shift in past:
                if len(frame):
                    dist = abs(frame.position.to_numpy(float)+shift-pos)
                    k = int(np.argmin(dist))
                    if dist[k] <= cfg.history_match_radius:
                        matches.append((float(dist[k]), float(frame.amplitude.iloc[k])))
            row.update(history_frame_count=len(past), history_support_count=len(matches),
                       history_support_fraction=len(matches)/len(past) if past else np.nan,
                       history_position_mad=float(np.median([v[0] for v in matches])) if matches else np.nan,
                       history_height_ratio=float(row["amplitude"]/max(np.median([v[1] for v in matches]), EPS)) if matches else np.nan)
        for name, ref in refs.items():
            gap = float(np.min(abs(ref-pos))) if len(ref) else np.nan
            row[f"reference_{name}_gap"] = gap
            row[f"reference_{name}_support"] = bool(gap <= cfg.stability_match_radius) if len(ref) else False
        rows.append(row)
    result = pd.DataFrame(rows, columns=columns)
    result.attrs.update(input_kind=kind, db_reference=db_reference,
                        coordinate_unit="sample bin" if axis is None else "same unit as axis",
                        feature_availability=dict(basic=True, template=template is not None,
                                                  frequency=data is not None, conditional_model=data is not None,
                                                  history=history is not None),
                        notes=["NaN denotes unavailable or geometrically undefined features.",
                               "Boundary peaks are height-filtered; their prominence widths are undefined.",
                               "Background contrast is not RF SNR; BIC gain is heuristic, not a calibrated probability.",
                               "History height ratios need consistent amplitude scale across frames."])
    return result


def extract_peak_windows(cir, peaks: pd.DataFrame, offsets, *, axis=None, input_kind="auto"):
    """Candidate-aligned amplitude/log-power windows for a shared CNN encoder.

    Returns amplitude, log_power_db, valid_mask and offsets. Amplitude is
    normalized by the full CIR maximum. Out-of-domain values are NaN, with an
    explicit validity mask. This function performs no model inference.
    """
    a, _, x, _, _, _, _ = _prepare(cir, axis, input_kind)
    offsets = np.asarray(offsets, float)
    if offsets.ndim != 1 or not np.isfinite(offsets).all():
        raise ValueError("offsets must be a finite 1-D array in axis units")
    coordinates = peaks.position.to_numpy(float)[:, None]+offsets[None, :]
    windows = np.interp(coordinates.ravel(), x, a, left=np.nan, right=np.nan).reshape(coordinates.shape)
    return dict(amplitude=windows, log_power_db=20*np.log10(np.maximum(windows, 1e-6)),
                valid_mask=np.isfinite(windows), offsets=offsets.copy())
