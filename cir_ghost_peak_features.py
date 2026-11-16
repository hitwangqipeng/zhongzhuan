"""Ghost/early-false-peak features, extending ``cir_peak_features.py``.

One input CIR -> one DataFrame row per candidate, with the original columns
followed by ``ghost_`` columns. No labels, truth distances, calibration offsets,
antenna fusion, peak selection or reliability probabilities are used.

Full complex CIR + FrequencyContext enables linear PSF explanation, conditional
residuals, frequency holdout, rewindowing and span-preserving tone dropout.
Magnitude/power/dB input retains geometry features; phase-dependent columns are
NaN. All coordinates, including baseline_position, use the units of ``axis``.

The midpoint/coefficient features assume a reciprocal *two-way* CFR H_tw=H**2.
They are diagnostics, not proof of a ghost or of the earliest physical path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from cir_peak_features import (
    FrequencyContext, PeakFeatureConfig, extract_peak_features, feature_schema,
)


@dataclass(frozen=True)
class GhostFeatureConfig:
    # None: derive N*axis_step/(max(active k)-min(active k)+1).
    # This is a bandwidth scale, NOT the zero-padded sample spacing.
    resolution: float | None = None
    max_reference_peaks: int = 6
    max_pair_peaks: int = 10
    reference_min_relative_height: float = 0.02
    reference_min_separation_resolution: float = 0.70
    candidate_exclusion_resolution: float = 0.45
    residual_radius_resolution: float = 0.65
    stability_radius_resolution: float = 0.55
    stability_min_relative_height: float = 0.005
    midpoint_tolerance_resolution: float = 0.20
    cv_repeats: int = 8
    cv_holdout_fraction: float = 0.25
    dropout_repeats: int = 8
    dropout_fraction: float = 0.10
    # Default: retain both observed band edges in every dropout/CV training view.
    preserve_band_edges: bool = True
    random_seed: int = 20261116


_GROUPS = {
    "geometry": {
        "ghost_resolution": "Nominal bandwidth scale in axis units; not FFT-grid resolution",
        "ghost_baseline_position": "Caller baseline, or strongest ROI position if omitted",
        "ghost_advance": "max(baseline_position-candidate_position,0), axis units",
        "ghost_advance_resolution": "Positive advance / nominal bandwidth scale",
        "ghost_front40_gap": "Candidate position minus own left 40%-amplitude crossing",
        "ghost_front40_gap_resolution": "Own 40%-edge gap / bandwidth scale",
        "ghost_front40_gap_to_width": "Own 40%-edge gap / half-power width",
        "ghost_later_stronger_count": "Other returned candidates later and at least as high",
        "ghost_later_stronger_gap_resolution": "Nearest later at-least-as-high candidate gap / scale",
    },
    "sampling": {
        "ghost_active_tone_count": "Number of declared measured tones",
        "ghost_missing_tone_fraction": "Missing FFT-grid tones within declared observed band span",
        "ghost_outside_search_power_fraction": "Full CIR power outside candidate search interval / total",
        "ghost_periodic_edge_power_fraction": "Power near both full-array ends within one bandwidth scale",
    },
    "conditional_model": {
        "ghost_reference_count": "Other amplitude-selected full-CIR peaks in conditional model",
        "ghost_reference_fit_nmse": "Raw-tone RSS without candidate / raw-tone energy",
        "ghost_joint_fit_nmse": "Raw-tone RSS with candidate / raw-tone energy",
        "ghost_fit_gain_fraction": "RSS decrease / RSS without candidate; in-sample diagnostic",
        "ghost_fit_gain_total_energy_fraction": "RSS decrease / observed raw-tone energy",
        "ghost_dictionary_condition": "Condition number with candidate; large means ambiguous fit",
        "ghost_explained_amplitude_ratio": "Other fitted components' amplitude here / observed amplitude",
        "ghost_explanation_phase_cosine": "Cosine of phase difference between explanation and observed response",
        "ghost_residual_amplitude_ratio": "Complex residual amplitude at candidate / original amplitude",
        "ghost_residual_energy_ratio": "Local residual CIR energy / local original CIR energy",
        "ghost_residual_peak_support": "A residual local maximum is within configured matching radius",
        "ghost_residual_peak_offset_resolution": "Signed offset of matched residual peak / scale",
        "ghost_residual_peak_relative_height": "Matched residual peak / strongest original ROI amplitude",
        "ghost_residual_peak_to_candidate": "Matched residual peak amplitude / observed candidate amplitude",
        "ghost_residual_prominence_ratio": "Matched residual amplitude prominence / strongest original ROI amplitude",
    },
    "frequency_holdout": {
        "ghost_cv_valid_splits": "Valid fixed-delay frequency-holdout fits",
        "ghost_cv_gain_mean": "Mean held-out RSS reduction / held-out RSS without candidate; can be negative",
        "ghost_cv_gain_median": "Median held-out conditional gain",
        "ghost_cv_gain_p10": "10th percentile held-out conditional gain",
        "ghost_cv_gain_min": "Worst held-out conditional gain",
        "ghost_cv_positive_fraction": "Fraction of valid splits with positive held-out conditional gain",
    },
    "stability": {
        "ghost_window_support_fraction": "Matched fraction across rectangular/Hann/Hamming raw-tone tapers",
        "ghost_window_offset_mad_resolution": "Median absolute matched peak offset / nominal scale",
        "ghost_window_relative_height_cv": "Matched within-view relative-height std/mean",
        "ghost_window_front40_offset_std_resolution": "Std of matched 40%-edge coordinate offsets / scale",
        "ghost_dropout_support_fraction": "Matched fraction after random interior-tone dropout",
        "ghost_dropout_offset_mad_resolution": "Median absolute matched peak offset / scale",
        "ghost_dropout_relative_height_cv": "Matched within-view relative-height std/mean",
        "ghost_dropout_front40_offset_std_resolution": "Std of matched 40%-edge coordinate offsets / scale",
    },
    "convolution": {
        "ghost_midpoint_gap": "Minimum |candidate-(parent1+parent2)/2| for bracketing reference pairs",
        "ghost_midpoint_gap_resolution": "Minimum midpoint mismatch / scale",
        "ghost_midpoint_pair_count": "Bracketing pairs within configured midpoint tolerance",
        "ghost_midpoint_parent_min_height": "Weaker parent amplitude / full-CIR maximum for nearest midpoint pair",
        "ghost_midpoint_parent_span_resolution": "Parent separation / scale for nearest midpoint pair",
        "ghost_midpoint_complex_error": "|b_c^2-4*b_i*b_j|/(|b_c^2|+|4*b_i*b_j|), nearest matching pair",
        "ghost_midpoint_log_amplitude_error": "Absolute log amplitude ratio of b_c^2 and 4*b_i*b_j",
        "ghost_midpoint_phase_error_pi": "Absolute phase difference of b_c^2 and 4*b_i*b_j / pi",
        "ghost_pair_dictionary_condition": "Condition number of fixed-delay dictionary used for pair coefficients",
        "ghost_pair_fit_nmse": "Raw-tone NMSE of joint parent-pool plus candidate fit",
    },
    "interactions": {
        "ghost_advance_explained_interaction": "Advance/scale * clipped explained-amplitude ratio; not a risk probability",
        "ghost_advance_window_instability": "Advance/scale * (1-window support fraction)",
        "ghost_advance_cv_disagreement": "Advance/scale * (1-held-out positive-gain fraction)",
    },
}


def ghost_feature_schema(config: PeakFeatureConfig | None = None, *, include_base=True) -> pd.DataFrame:
    """Same schema format as feature_schema(); every ghost column is documented."""
    rows = []
    for group, fields in _GROUPS.items():
        needs = ("CIR; normalized geometry also needs resolution or FrequencyContext"
                 if group == "geometry" else "full complex CIR + FrequencyContext")
        rows.extend(dict(feature=k, group=group, requires=needs, description=v)
                    for k, v in fields.items())
    extra = pd.DataFrame(rows)
    return pd.concat([feature_schema(config), extra], ignore_index=True) if include_base else extra


def _validate(cfg):
    for name in ("max_reference_peaks", "max_pair_peaks", "cv_repeats", "dropout_repeats"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("reference_min_relative_height", "stability_min_relative_height"):
        value = getattr(cfg, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0,1]")
    for name in ("cv_holdout_fraction", "dropout_fraction"):
        value = getattr(cfg, name)
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be finite and in (0,1)")
    for name in ("reference_min_separation_resolution", "candidate_exclusion_resolution",
                 "residual_radius_resolution", "stability_radius_resolution", "midpoint_tolerance_resolution"):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg.resolution is not None and (not np.isfinite(cfg.resolution) or cfg.resolution <= 0):
        raise ValueError("resolution must be positive and finite or None")
    if not isinstance(cfg.preserve_band_edges, (bool, np.bool_)):
        raise ValueError("preserve_band_edges must be boolean")
    if not isinstance(cfg.random_seed, (int, np.integer)) or cfg.random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")


def _prepare_amplitude(cir, kind):
    if kind == "complex":
        return abs(np.asarray(cir))
    a = np.asarray(cir, float)
    if kind.endswith("_db"):
        return 10.**((a-a.max())/20.)
    return np.sqrt(a) if kind == "power" else a


def _refine(a, peak, enabled=True):
    n = len(a)
    left, center, right = a[(peak-1) % n], a[peak], a[(peak+1) % n]
    denominator = left-2*center+right
    if enabled and denominator < -np.finfo(float).eps and center >= max(left, right):
        return float((peak+np.clip(.5*(left-right)/denominator, -.5, .5)) % n)
    return float(peak)


def _peaks(a):
    # Circular neighbours prevent a cut search-window endpoint becoming a peak.
    p = find_peaks(np.r_[a[-1], a, a[0]])[0]-1
    return p[(p >= 0) & (p < len(a))]


def _periodic_delta(a, b, period):
    return (np.asarray(a)-b+period/2) % period-period/2


def _basis(k, positions, n):
    return np.exp(-2j*np.pi*np.outer(k, positions)/n)


def _fit(a, z):
    if a.shape[1] == 0:
        return np.empty(0, complex), np.zeros_like(z)
    coefficients = np.linalg.lstsq(a, z, rcond=1e-8)[0]
    return coefficients, a @ coefficients


def _power(z):
    return float(np.vdot(z, z).real)


def _safe_ratio(a, b, floor):
    return float(a/max(float(b), floor))


def _tone_view(z, weights, bins, n, shifted):
    spectrum = np.zeros(n, complex)
    spectrum[bins] = z*weights
    h = np.fft.ifft(spectrum)
    return np.fft.fftshift(h) if shifted else h


def _edge40(a, sample):
    # Walk to the nearest left crossing of 40% of this local peak amplitude.
    n = len(a); level = .4*a[sample]
    for step in range(1, n):
        j = (sample-step) % n
        if a[j] <= level:
            prev = (j+1) % n
            fraction = (level-a[j])/max(a[prev]-a[j], np.finfo(float).tiny)
            return sample-step+fraction
    return np.nan


def _psf_width(weights, k, n):
    # FWHM on a dense positive-delay grid. Side-lobe amplitudes do not set width.
    nominal = n/(k[-1]-k[0]+1)
    grid = np.linspace(0, 3*nominal, 257)
    amplitude = abs(np.exp(2j*np.pi*np.outer(grid, k)/n) @ weights)
    if amplitude[0] <= 0:
        return nominal
    power = (amplitude/amplitude[0])**2
    ids = np.flatnonzero(power <= .5)
    if not len(ids):
        return nominal
    j = ids[0]
    half = np.interp(.5, power[j-1:j+1][::-1], grid[j-1:j+1][::-1])
    return float(2*half)


def _reference_pool(a, x, dx, resolution, cfg, refine, n):
    peaks = _peaks(a)
    if not len(peaks) or a.max() <= 0:
        return []
    ordered = sorted(peaks, key=lambda p: (-a[p], int(p)))
    selected = []
    for p in ordered:
        if a[p] < cfg.reference_min_relative_height*a.max():
            continue
        sample = _refine(a, int(p), refine)
        # Keep x's interval convention, including translated or fftshifted axes.
        position = float(x[0]+sample*dx)
        if all(abs(_periodic_delta(position, q[1], n*dx)) >=
               cfg.reference_min_separation_resolution*resolution for q in selected):
            selected.append((sample, position, float(a[p]/a.max())))
    return selected


def _holdouts(m, repeats, fraction, seed, preserve_edges, *, stratified):
    rng = np.random.default_rng(seed)
    available = np.arange(1, m-1) if preserve_edges else np.arange(m)
    if not len(available):
        return []
    count = max(1, min(len(available), round(m*fraction)))
    blocks = np.array_split(available, min(4, count))
    result = []
    for _ in range(repeats):
        if stratified:
            # One held-out point per broad band, then fill remaining slots.
            initial = np.asarray([rng.choice(b) for b in blocks if len(b)], int)
            rest = np.setdiff1d(available, initial)
            chosen = np.r_[initial, rng.choice(rest, count-len(initial), replace=False)]
        else:
            chosen = rng.choice(available, count, replace=False)
        result.append(np.sort(chosen))
    return result


def _make_views(data, cfg, dx):
    z, k, bins, n, shifted = (data[v] for v in ("z", "k", "bins", "n", "shifted"))
    # Tapers follow actual FFT frequency indices, including missing tone gaps.
    span = int(round(k[-1]-k[0]))+1
    within = np.rint(k-k[0]).astype(int)
    windows = [np.ones(len(k)), np.hanning(span)[within], np.hamming(span)[within]]
    families = []
    for weights in windows:
        families.append((_tone_view(z, weights, bins, n, shifted),
                         _psf_width(weights, k, n)*dx))
    dropout = []
    for omitted in _holdouts(len(k), cfg.dropout_repeats, cfg.dropout_fraction,
                             cfg.random_seed+1, cfg.preserve_band_edges, stratified=False):
        weights = data["weights"].copy(); weights[omitted] = 0
        if np.count_nonzero(weights) < 4:
            continue
        dropout.append((_tone_view(z, weights, bins, n, shifted),
                        _psf_width(weights, k, n)*dx))
    return families, dropout


def _stability(position, views, x, dx, resolution, cfg, roi):
    offsets, heights, edges = [], [], []
    n = len(x); period = n*dx
    for h, width in views:
        a = abs(h); maximum = float(a[roi].max())
        pp = _peaks(a)
        pp = pp[roi[pp] & (a[pp] >= cfg.stability_min_relative_height*maximum)]
        if not len(pp) or maximum <= 0:
            continue
        samples = np.asarray([_refine(a, int(p)) for p in pp])
        positions = x[0]+samples*dx
        delta = _periodic_delta(positions, position, period)
        j = int(np.argmin(abs(delta)))
        radius = cfg.stability_radius_resolution*max(resolution, width)
        if abs(delta[j]) <= radius:
            offsets.append(float(delta[j])); heights.append(float(a[pp[j]]/maximum))
            edge_sample = _edge40(a, int(pp[j]))
            if np.isfinite(edge_sample):
                edges.append(float(_periodic_delta(x[0]+edge_sample*dx, position, period)))
    return dict(support_fraction=len(offsets)/len(views) if views else np.nan,
                offset_mad_resolution=float(np.median(abs(np.asarray(offsets)))/resolution) if offsets else np.nan,
                relative_height_cv=float(np.std(heights)/np.mean(heights)) if heights else np.nan,
                front40_offset_std_resolution=float(np.std(edges)/resolution) if len(edges) >= 2 else np.nan)


def _conditional(data, sample, position, refs, x, dx, resolution, cfg, roi, holdouts):
    z, k, weights, n = (data[v] for v in ("z", "k", "weights", "n"))
    shift = n//2 if data["shifted"] else 0
    s0 = [r[0]-shift for r in refs]
    a0 = _basis(k, s0, n)
    ac = _basis(k, [sample-shift], n)
    a1 = np.column_stack([a0, ac])
    _, f0 = _fit(a0, z); _, f1 = _fit(a1, z)
    residual = z-f0
    energy = _power(z); floor = max(energy*1e-14, np.finfo(float).tiny)
    rss0, rss1 = _power(residual), _power(z-f1)
    gain = max(0., rss0-rss1)
    steer = np.exp(2j*np.pi*k*(sample-shift)/n)*weights/n
    actual = np.dot(steer, z); explained = np.dot(steer, f0)
    here = actual-explained
    amp_floor = max(np.sqrt(energy)*1e-12/n, np.finfo(float).tiny)
    hr = _tone_view(residual, weights, data["bins"], n, data["shifted"])
    original = data["h"]
    delta = _periodic_delta(x, position, n*dx)
    near = abs(delta) <= cfg.residual_radius_resolution*resolution
    row = dict(ghost_reference_count=len(refs), ghost_reference_fit_nmse=rss0/max(energy, floor),
               ghost_joint_fit_nmse=rss1/max(energy, floor),
               ghost_fit_gain_fraction=gain/max(rss0, floor),
               ghost_fit_gain_total_energy_fraction=gain/max(energy, floor),
               ghost_dictionary_condition=float(np.linalg.cond(a1)),
               ghost_explained_amplitude_ratio=abs(explained)/max(abs(actual), amp_floor),
               ghost_explanation_phase_cosine=float(np.cos(np.angle(explained*np.conj(actual))))
               if abs(explained) > amp_floor and abs(actual) > amp_floor else np.nan,
               ghost_residual_amplitude_ratio=abs(here)/max(abs(actual), amp_floor),
               ghost_residual_energy_ratio=_safe_ratio(_power(hr[near]), _power(original[near]), floor/n**2),
               ghost_residual_peak_support=0.)
    pp = _peaks(abs(hr))
    candidates = pp[abs(delta[pp]) <= cfg.residual_radius_resolution*resolution]
    # A near-zero numerical residual is not a supported physical component.
    candidates = candidates[abs(hr[candidates]) > max(abs(original[roi]).max()*1e-10, amp_floor)]
    if len(candidates):
        p = int(candidates[np.argmin(abs(delta[candidates]))])
        s = _refine(abs(hr), p)
        offset = float(_periodic_delta(x[0]+s*dx, position, n*dx))
        wrapped = np.r_[abs(hr[-1]), abs(hr), abs(hr[0])]
        prominence = float(peak_prominences(wrapped, [p+1])[0][0])
        original_max = max(float(abs(original[roi]).max()), amp_floor)
        row.update(ghost_residual_peak_support=1., ghost_residual_peak_offset_resolution=offset/resolution,
                   ghost_residual_peak_relative_height=float(abs(hr[p])/original_max),
                   ghost_residual_peak_to_candidate=float(abs(hr[p])/max(abs(actual), amp_floor)),
                   ghost_residual_prominence_ratio=prominence/original_max)
    gains = []
    for test in holdouts:
        train = np.setdiff1d(np.arange(len(z)), test)
        if len(train) < 2*a1.shape[1] or not len(test):
            continue
        c0, _ = _fit(a0[train], z[train]); c1, _ = _fit(a1[train], z[train])
        r0 = z[test]-a0[test] @ c0; r1 = z[test]-a1[test] @ c1
        denominator = _power(r0)
        # Null denominators do not create fabricated perfect CV scores.
        if denominator <= floor:
            continue
        gains.append((_power(r0)-_power(r1))/denominator)
    row["ghost_cv_valid_splits"] = len(gains)
    if gains:
        row.update(ghost_cv_gain_mean=float(np.mean(gains)), ghost_cv_gain_median=float(np.median(gains)),
                   ghost_cv_gain_p10=float(np.quantile(gains, .1)), ghost_cv_gain_min=float(np.min(gains)),
                   ghost_cv_positive_fraction=float(np.mean(np.asarray(gains) > 0)))
    return row


def _pair_features(data, sample, position, parents, resolution, cfg):
    row = {"ghost_midpoint_pair_count": 0.}
    pairs = [(i, j, abs(position-(a[1]+b[1])/2))
             for (i, a), (j, b) in combinations(enumerate(parents), 2)
             if min(a[1], b[1]) < position < max(a[1], b[1])]
    if not pairs:
        return row
    i, j, gap = min(pairs, key=lambda p: (p[2], -min(parents[p[0]][2], parents[p[1]][2])))
    row.update(ghost_midpoint_gap=gap, ghost_midpoint_gap_resolution=gap/resolution,
               ghost_midpoint_pair_count=sum(g <= cfg.midpoint_tolerance_resolution*resolution for _, _, g in pairs),
               ghost_midpoint_parent_min_height=min(parents[i][2], parents[j][2]),
               ghost_midpoint_parent_span_resolution=abs(parents[i][1]-parents[j][1])/resolution)
    if data is None or gap > cfg.midpoint_tolerance_resolution*resolution:
        return row
    n, k, z = data["n"], data["k"], data["z"]
    shift = n//2 if data["shifted"] else 0
    positions = [p[0]-shift for p in parents]+[sample-shift]
    a = _basis(k, positions, n)
    coefficients, fitted = _fit(a, z)
    lhs = coefficients[-1]**2; rhs = 4*coefficients[i]*coefficients[j]
    energy = _power(z); floor = max(energy/len(z)*1e-14, np.finfo(float).tiny)
    row.update(ghost_pair_dictionary_condition=float(np.linalg.cond(a)),
               ghost_pair_fit_nmse=_power(z-fitted)/max(energy, floor))
    if abs(lhs) <= floor or abs(rhs) <= floor:
        return row
    row.update(ghost_midpoint_complex_error=float(abs(lhs-rhs)/(abs(lhs)+abs(rhs))),
               ghost_midpoint_log_amplitude_error=float(abs(np.log(abs(lhs)/abs(rhs)))),
               ghost_midpoint_phase_error_pi=float(abs(np.angle(lhs*np.conj(rhs)))/np.pi))
    return row


def extract_ghost_peak_features(
    cir, axis=None, *, input_kind="auto", config: PeakFeatureConfig | None = None,
    ghost_config: GhostFeatureConfig | None = None, peak_indices: Sequence[int] | None = None,
    frequency_context: FrequencyContext | None = None, baseline_position: float | None = None,
    psf=None, history: Sequence[pd.DataFrame] | None = None,
    history_shifts: Sequence[float] | None = None,
    reference_peak_sets: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Extract base + ghost features for every detected or supplied candidate.

    ``peak_indices`` and all other common arguments match extract_peak_features.
    References are independently selected from the FULL CIR, not just supplied
    candidates or a cropped search interval. No bias subtraction is applied.

    Frequency holdout refits complex coefficients on training tones but uses
    candidate/reference delays detected from this same full observation. It is
    a fixed-dictionary diagnostic, not an independent generalization estimate.
    """
    cfg, gc = config or PeakFeatureConfig(), ghost_config or GhostFeatureConfig()
    _validate(gc)
    if baseline_position is not None and not np.isfinite(baseline_position):
        raise ValueError("baseline_position must be finite or None")
    result = extract_peak_features(cir, axis, input_kind=input_kind, config=cfg,
                                  peak_indices=peak_indices, frequency_context=frequency_context, psf=psf,
                                  history=history, history_shifts=history_shifts,
                                  reference_peak_sets=reference_peak_sets)
    attrs = result.attrs.copy()
    names = list(ghost_feature_schema(include_base=False).feature)
    extra = pd.DataFrame(np.nan, index=result.index, columns=names)
    a = _prepare_amplitude(cir, attrs["input_kind"])
    a = a/max(float(a.max()), np.finfo(float).tiny)
    n = len(a); x = np.arange(n, dtype=float) if axis is None else np.asarray(axis, float)
    dx = float(x[1]-x[0]); resolution = gc.resolution
    roi = np.ones(n, bool) if cfg.search_range is None else (x >= cfg.search_range[0]) & (x <= cfg.search_range[1])
    data = None; views = None; holdouts = []
    if frequency_context is not None:
        bins = np.asarray(frequency_context.active_bins, int)
        k = np.rint(np.fft.fftfreq(n)[bins]*n)
        weights = np.ones(len(bins)) if frequency_context.weights is None else np.asarray(frequency_context.weights, float)
        # Normalize before fitting to make all relative features gain invariant.
        h = np.asarray(cir, complex)/max(float(abs(np.asarray(cir)).max()), np.finfo(float).tiny)
        spectrum = np.fft.fft(np.fft.ifftshift(h) if frequency_context.fft_shifted else h)
        z = spectrum[bins]/weights
        if resolution is None:
            resolution = n*dx/(k[-1]-k[0]+1)
        data = dict(n=n, bins=bins, k=k, weights=weights, z=z, h=h, shifted=frequency_context.fft_shifted)
        views = _make_views(data, gc, dx) if len(result) else None
        holdouts = _holdouts(len(z), gc.cv_repeats, gc.cv_holdout_fraction,
                             gc.random_seed, gc.preserve_band_edges, stratified=True)
    known_resolution = resolution is not None
    pool = _reference_pool(a, x, dx, resolution, gc, cfg.refine_positions, n) if known_resolution else []
    for index, base in result.iterrows():
        position = float(base.position)
        baseline = float(base.global_strongest_position if baseline_position is None else baseline_position)
        advance = max(baseline-position, 0.)
        edge_gap = position-base.left_40pct_position
        later = result[(result.position > position) & (result.amplitude >= base.amplitude)]
        row = dict(ghost_resolution=resolution if known_resolution else np.nan,
                   ghost_baseline_position=baseline, ghost_advance=advance,
                   ghost_front40_gap=edge_gap,
                   ghost_front40_gap_to_width=edge_gap/base.half_power_width
                   if np.isfinite(base.half_power_width) and base.half_power_width > 0 else np.nan,
                   ghost_later_stronger_count=len(later))
        if known_resolution:
            row.update(ghost_advance_resolution=advance/resolution,
                       ghost_front40_gap_resolution=edge_gap/resolution,
                       ghost_later_stronger_gap_resolution=float((later.position.min()-position)/resolution)
                       if len(later) else np.nan)
            sample = (position-x[0])/dx
            eligible = [p for p in pool if abs(_periodic_delta(p[1], position, n*dx)) >
                        gc.candidate_exclusion_resolution*resolution]
            pair_limit = gc.max_pair_peaks if data is None else min(gc.max_pair_peaks, max(2, len(data["z"])//3-1))
            parents = eligible[:pair_limit]
            row.update(_pair_features(data, sample, position, parents, resolution, gc))
        if data is not None:
            m = len(data["z"])
            min_train = min((m-len(v) for v in holdouts), default=m)
            limit = min(gc.max_reference_peaks, max(0, min_train//2-1))
            refs = eligible[:limit]
            row.update(_conditional(data, sample, position, refs, x, dx, resolution, gc, roi, holdouts))
            row.update(ghost_active_tone_count=m,
                       ghost_missing_tone_fraction=1-m/(data["k"][-1]-data["k"][0]+1),
                       ghost_outside_search_power_fraction=_safe_ratio(_power(data["h"][~roi]), _power(data["h"]), np.finfo(float).tiny),
                       ghost_periodic_edge_power_fraction=_safe_ratio(
                           _power(data["h"][(x-x[0] < resolution) | (x[-1]-x < resolution)]),
                           _power(data["h"]), np.finfo(float).tiny))
            for prefix, family in zip(("window", "dropout"), views):
                summary = _stability(position, family, x, dx, resolution, gc, roi)
                row.update({f"ghost_{prefix}_{name}": value for name, value in summary.items()})
            row["ghost_advance_explained_interaction"] = advance/resolution*np.clip(row["ghost_explained_amplitude_ratio"], 0, 1)
            row["ghost_advance_window_instability"] = advance/resolution*(1-row["ghost_window_support_fraction"])
            row["ghost_advance_cv_disagreement"] = advance/resolution*(1-row.get("ghost_cv_positive_fraction", np.nan))
        for key, value in row.items():
            extra.at[index, key] = value
    # Infinite condition numbers are represented by missing values, not infinities.
    extra.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined = pd.concat([result, extra], axis=1)
    combined.attrs = attrs
    combined.attrs.update(ghost_config=asdict(gc), ghost_feature_version="1.0",
                          ghost_feature_availability=dict(geometry=True, bandwidth_scale=known_resolution,
                                                          complex_model=data is not None, frequency_holdout=data is not None,
                                                          stability=data is not None, convolution_geometry=known_resolution,
                                                          convolution_coefficients=data is not None),
                          ghost_notes=[
                              "No labels, distance calibration or final peak choice are produced.",
                              "All coordinates must use the same uncorrected/calibrated axis convention.",
                              "Fixed-delay fits may be ill-conditioned; condition numbers are diagnostic.",
                              "Holdout uses delays selected from this observation; it is not independent validation.",
                              "Hann/Hamming may merge nearby paths; stability is supporting evidence only.",
                              "References outside search_range can explain in-range sidelobes.",
                              "Midpoint coefficient relation assumes reciprocal two-way H_tw=H^2 and separable components.",
                              "Full-array edge power does not prove periodic aliasing.",
                              "NaN denotes unavailable, degenerate or geometrically undefined features.",
                          ])
    return combined
