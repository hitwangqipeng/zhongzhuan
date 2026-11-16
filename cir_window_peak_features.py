"""Cross-window anti-sidelobe features extending ``cir_peak_features.py``.

One CIR -> one DataFrame row per original candidate. Original columns are
preserved; ``win_`` columns are appended. No reliability labels, range offsets,
path identifiers, candidate rejection or final peak selection are produced.

Rewindowing requires a FULL complex CIR and FrequencyContext. Raw measured
tones are recovered before applying each new taper; an existing taper is not
multiplied a second time. Windows are defined on the declared frequency span,
including missing tones, and are corrected for their measured coherent gain.
The comparison baseline is the rectangular reconstruction of recovered tones.

Fixed-coordinate measurements and matched-local-peak measurements are separate.
Missing matches are NaN plus a support flag, never a distant substituted peak.
These correlated views of the same IQ are evidence features, not independent
measurements or calibrated probabilities of a physical path.

For the recommended six-feature subset, use extract_compact_window_peak_features.
It uses Hann only, disables template fitting and returns candidate identifiers
plus six model features. The full interface remains available for ablation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths, windows as sp_windows

from cir_peak_features import (
    FrequencyContext, PeakFeatureConfig, extract_peak_features, feature_schema,
)


@dataclass(frozen=True)
class WindowFeatureConfig:
    windows: tuple[str, ...] = ("hann", "kaiser")
    sampling: str = "periodic"
    kaiser_beta: float = 4.0
    resolution: float | None = None
    derivative_step_resolution: float = 0.15
    match_radius_resolution: float = 0.60
    max_reference_peaks: int = 4
    reference_min_separation_resolution: float = 0.70
    candidate_exclusion_resolution: float = 0.55
    include_template_features: bool = True


# Physical/redundancy-based starting set, not an empirically ranked importance list.
# Keep the normalized response requested by the caller, plus the gain-corrected
# response to distinguish local attenuation from a moving global maximum.
COMPACT_WINDOW_FEATURES = (
    "win_hann_relative_height_log_ratio",
    "win_hann_fixed_amp_log_ratio",
    "win_hann_peak_support",
    "win_hann_peak_offset_resolution",
    "win_hann_phase_cosine",
    "win_hann_right_valley_fill",
)
COMPACT_WINDOW_IDENTIFIERS = ("peak_index", "position")


_SHARED = {
    "win_resolution": "Declared/derived bandwidth scale in axis units, not zero-padded bin spacing",
    "win_left_valley_gap_resolution": "Rectangular-view nearest left valley gap / bandwidth scale",
    "win_right_valley_gap_resolution": "Rectangular-view nearest right valley gap / bandwidth scale",
    "win_valley_bounds_available": "Both rectangular-view bracketing valleys exist",
    "win_peak_support_fraction": "Fraction of new windows with a peak inside original valley basin and matching radius",
    "win_peak_offset_mad_resolution": "Median absolute matched peak displacement / scale; NaN if none",
    "win_fixed_amp_log_ratio_std": "Std of fixed-coordinate corrected-amplitude log ratios, dB",
    "win_self_template_nmse": "Complex multiwindow vector NMSE against a scaled constant (isolated component at candidate)",
    "win_sidelobe_template_nmse": "Best multiwindow NMSE explained by one other full-CIR strong peak",
    "win_sidelobe_minus_self_nmse": "Sidelobe hypothesis NMSE minus self hypothesis NMSE; evidence, not probability",
    "win_sidelobe_reference_gap_resolution": "Signed candidate-minus-reference displacement / scale for best explanation",
    "win_sidelobe_reference_relative_height": "Best explaining reference's rectangular amplitude / ROI maximum",
}

_PER_WINDOW = {
    "coherent_gain": "Sum of new taper on measured tones / measured tone count",
    "fixed_amp_log_ratio": "20log10(corrected window amplitude / rectangular amplitude) at original candidate",
    "global_max_log_ratio": "20log10(corrected window ROI maximum / rectangular ROI maximum)",
    "relative_height": "Fixed-coordinate corrected amplitude / this window's ROI maximum",
    "relative_height_change": "Fixed-coordinate within-view relative height minus rectangular relative height",
    "relative_height_log_ratio": "fixed_amp_log_ratio - global_max_log_ratio; separates normalization effect",
    "amplitude_rank_change": "Within-original-candidate amplitude-rank change / max(candidate count-1,1)",
    "fixed_slope": "Symmetric amplitude difference at +/- derivative step / center amplitude",
    "slope_change": "Fixed-coordinate slope minus rectangular slope",
    "fixed_curvature": "(2*center-left-right) / center amplitude at derivative step",
    "curvature_change": "Fixed-coordinate curvature minus rectangular curvature",
    "phase_cosine": "Cosine of window-versus-rectangular complex phase difference at original candidate",
    "phase_sine": "Sine of window-versus-rectangular complex phase difference; NaN at numerically null amplitudes",
    "left_valley_fill": "Window amplitude at original left valley / window amplitude at candidate",
    "right_valley_fill": "Window amplitude at original right valley / window amplitude at candidate",
    "peak_support": "A windowed local maximum matches within original basin and configured radius, no height cut",
    "peak_offset_resolution": "Signed matched-local-peak displacement / bandwidth scale",
    "matched_amp_log_ratio": "Corrected matched-peak amplitude versus rectangular fixed-candidate amplitude, dB",
    "prominence_log_ratio": "Matched amplitude prominence versus rectangular candidate prominence, dB",
    "matched_width_resolution": "Matched half-amplitude-prominence width / bandwidth scale",
    "matched_width_ratio": "Matched width / rectangular candidate half-prominence width",
    "basin_candidate_count": "Original returned candidates within the window's bracketing-valley basin around this position",
}


def _base_config(config):
    # Do not introduce an amplitude decision threshold in this extension.
    return config if config is not None else PeakFeatureConfig(min_relative_height=0., min_relative_prominence=0.)


def _validate(cfg):
    allowed = {"hann", "hamming", "kaiser", "blackman"}
    if not cfg.windows or len(set(cfg.windows)) != len(cfg.windows) or any(w not in allowed for w in cfg.windows):
        raise ValueError("windows must be distinct names from hann/hamming/kaiser/blackman")
    if cfg.sampling not in {"periodic", "symmetric"}:
        raise ValueError("sampling must be periodic or symmetric")
    if not np.isfinite(cfg.kaiser_beta) or cfg.kaiser_beta <= 0:
        raise ValueError("kaiser_beta must be finite and positive")
    if cfg.resolution is not None and (not np.isfinite(cfg.resolution) or cfg.resolution <= 0):
        raise ValueError("resolution must be positive and finite or None")
    for name in ("derivative_step_resolution", "match_radius_resolution", "reference_min_separation_resolution", "candidate_exclusion_resolution"):
        if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if isinstance(cfg.max_reference_peaks, bool) or not isinstance(cfg.max_reference_peaks, (int, np.integer)) or cfg.max_reference_peaks < 1:
        raise ValueError("max_reference_peaks must be a positive integer")
    if not isinstance(cfg.include_template_features, (bool, np.bool_)):
        raise ValueError("include_template_features must be boolean")


def window_feature_schema(config: PeakFeatureConfig | None = None, *,
                          window_config: WindowFeatureConfig | None = None,
                          include_base=True) -> pd.DataFrame:
    """Same feature/group/requires/description columns as feature_schema()."""
    wc = window_config or WindowFeatureConfig(); _validate(wc)
    rows = [dict(feature=k, group="cross_window", requires="full complex CIR + FrequencyContext", description=v)
            for k, v in _SHARED.items()]
    for name in wc.windows:
        rows.extend(dict(feature=f"win_{name}_{k}", group=f"window_{name}",
                         requires="full complex CIR + FrequencyContext", description=v)
                    for k, v in _PER_WINDOW.items())
    extra = pd.DataFrame(rows)
    return pd.concat([feature_schema(_base_config(config)), extra], ignore_index=True) if include_base else extra


def compact_window_feature_schema(config: PeakFeatureConfig | None = None, *,
                                  include_base: bool = False) -> pd.DataFrame:
    """Schema for identifiers + six features, or base columns + six features.

    peak_index and position identify candidates; do not automatically treat
    these identifiers as model inputs. COMPACT_WINDOW_FEATURES lists inputs.
    """
    full = window_feature_schema(config, window_config=WindowFeatureConfig(windows=("hann",)))
    base_columns = list(feature_schema(_base_config(config)).feature) if include_base else list(COMPACT_WINDOW_IDENTIFIERS)
    return full.set_index("feature", drop=False).loc[base_columns + list(COMPACT_WINDOW_FEATURES)].reset_index(drop=True)


def select_compact_window_features(features: pd.DataFrame, *,
                                   include_base: bool = False) -> pd.DataFrame:
    """Select six representative inputs from an existing full feature table.

    Requires Hann columns. Values and original candidate rows are unchanged.
    include_base=True retains all original non-win_ columns for existing models.
    """
    base_columns = [c for c in features if not c.startswith("win_")] if include_base else list(COMPACT_WINDOW_IDENTIFIERS)
    columns = base_columns + list(COMPACT_WINDOW_FEATURES)
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"Compact selection requires Hann features and candidate identifiers; missing: {missing}")
    result = features.loc[:, columns].copy()
    result.attrs = features.attrs.copy()
    result.attrs.update(window_feature_subset="compact_six",
                        window_model_features=list(COMPACT_WINDOW_FEATURES),
                        window_selection_basis="physical interpretation and redundancy; not validated real-car importance")
    return result


def extract_compact_window_peak_features(
    cir, axis=None, *, input_kind="auto", config: PeakFeatureConfig | None = None,
    window_config: WindowFeatureConfig | None = None, peak_indices: Sequence[int] | None = None,
    frequency_context: FrequencyContext | None = None, psf=None,
    history: Sequence[pd.DataFrame] | None = None, history_shifts: Sequence[float] | None = None,
    reference_peak_sets: Mapping[str, Sequence[float]] | None = None,
    include_base: bool = False,
) -> pd.DataFrame:
    """Recommended small set: Hann only, six additional model features.

    Default output is peak_index, position and COMPACT_WINDOW_FEATURES.
    include_base=True instead retains all existing base columns. No candidate
    is removed. A missing matched peak has support=0 and displacement=NaN.
    window_config may customize matching/scale, but must use Hann only and
    include_template_features=False to avoid unused extra calculations.
    """
    wc = window_config or WindowFeatureConfig(windows=("hann",), include_template_features=False)
    _validate(wc)
    if tuple(wc.windows) != ("hann",) or wc.include_template_features:
        raise ValueError("Compact extraction requires windows=('hann',) and include_template_features=False")
    full = extract_window_peak_features(cir, axis, input_kind=input_kind, config=config,
        window_config=wc, peak_indices=peak_indices, frequency_context=frequency_context,
        psf=psf, history=history, history_shifts=history_shifts, reference_peak_sets=reference_peak_sets)
    return select_compact_window_features(full, include_base=include_base)


def _peaks(a):
    p = find_peaks(np.r_[a[-1], a, a[0]])[0]-1
    return p[(p >= 0) & (p < len(a))]


def _refine(a, p):
    n = len(a); left, center, right = a[(p-1) % n], a[p], a[(p+1) % n]
    den = left-2*center+right
    if den < 0 and center >= max(left, right):
        return float((p+np.clip(.5*(left-right)/den, -.5, .5)) % n)
    return float(p)


def _delta(a, b, n):
    return (np.asarray(a)-b+n/2) % n-n/2


def _valleys(a):
    return _peaks(-a).astype(float)


def _bounds(valleys, sample, n):
    delta = _delta(valleys, sample, n)
    left, right = delta[delta < -1e-9], delta[delta > 1e-9]
    return (float(left.max()) if len(left) else np.nan,
            float(right.min()) if len(right) else np.nan)


def _log_ratio(a, b, floor):
    return float(20*np.log10((a+floor)/(b+floor))) if b > floor else np.nan


def _ratio(a, b, floor):
    return float(a/b) if b > floor else np.nan


def _shape(a):
    p = _peaks(a)
    if not len(p):return dict(peaks=p, samples=np.empty(0), prominence=np.empty(0), widths=np.empty(0), valleys=_valleys(a))
    # Full neighboring periods preserve prominence/width at the FFT wrap point.
    wrapped = np.tile(a, 3)
    prominence = peak_prominences(wrapped, p+len(a))
    widths = peak_widths(wrapped, p+len(a), rel_height=.5, prominence_data=prominence)[0]
    return dict(peaks=p, samples=np.array([_refine(a, int(v)) for v in p]),
                prominence=prominence[0], widths=widths, valleys=_valleys(a))


def _data(cir, context):
    n = len(cir); scale = float(abs(np.asarray(cir)).max())
    h = np.asarray(cir, complex)/max(scale, np.finfo(float).tiny)
    spectrum = np.fft.fft(np.fft.ifftshift(h) if context.fft_shifted else h)
    bins = np.asarray(context.active_bins, int)
    k = np.rint(np.fft.fftfreq(n)[bins]*n).astype(int)
    applied = np.ones(len(bins)) if context.weights is None else np.asarray(context.weights, float)
    z = spectrum[bins]/applied
    return dict(n=n, bins=bins, k=k, z=z, shifted=context.fft_shifted)


def _taper(name, data, cfg):
    k = data["k"]; span = int(k[-1]-k[0]+1); ids = k-k[0]
    argument = ("kaiser", cfg.kaiser_beta) if name == "kaiser" else name
    full = sp_windows.get_window(argument, span, fftbins=cfg.sampling == "periodic")
    return full[ids]


def _view(data, weights):
    n, m = data["n"], len(data["z"])
    if weights.sum() <= np.finfo(float).tiny:return None
    spectrum = np.zeros(n, complex); spectrum[data["bins"]] = data["z"]*weights*m/weights.sum()
    h = np.fft.ifft(spectrum)
    return np.fft.fftshift(h) if data["shifted"] else h


def _evaluate(data, weights, samples, *, template=False):
    samples = np.atleast_1d(samples).astype(float)
    if data["shifted"]:samples = samples-data["n"]//2
    tone = np.ones(len(weights)) if template else data["z"]
    return (np.exp(2j*np.pi*np.outer(samples, data["k"])/data["n"]) @ (tone*weights)
            *len(weights)/(data["n"]*weights.sum()))


def _nmse(target, template):
    denominator = float(np.vdot(template, template).real)
    energy = float(np.vdot(target, target).real)
    if denominator <= np.finfo(float).tiny or energy <= np.finfo(float).tiny:return np.nan
    coefficient = np.vdot(template, target)/denominator
    return float(np.sum(abs(target-coefficient*template)**2)/energy)


def extract_window_peak_features(
    cir, axis=None, *, input_kind="auto", config: PeakFeatureConfig | None = None,
    window_config: WindowFeatureConfig | None = None, peak_indices: Sequence[int] | None = None,
    frequency_context: FrequencyContext | None = None, psf=None,
    history: Sequence[pd.DataFrame] | None = None, history_shifts: Sequence[float] | None = None,
    reference_peak_sets: Mapping[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Append documented cross-window features to the unchanged base features.

    With no config, height/prominence thresholds are zero. Passing an explicit
    config retains its candidate policy; peak_indices bypasses those thresholds
    exactly as in extract_peak_features(). Windows never remove returned rows.

    If an input taper has already been applied, FrequencyContext.weights must
    describe its positive measured weights. Unrecoverable zero-weight tones
    must be absent from active_bins, not divided by zero or invented.
    """
    cfg, wc = _base_config(config), window_config or WindowFeatureConfig(); _validate(wc)
    base = extract_peak_features(cir, axis, input_kind=input_kind, config=cfg, peak_indices=peak_indices,
                                 frequency_context=frequency_context, psf=psf, history=history,
                                 history_shifts=history_shifts, reference_peak_sets=reference_peak_sets)
    extra = pd.DataFrame(np.nan, index=base.index, columns=list(window_feature_schema(window_config=wc, include_base=False).feature))
    attrs = base.attrs.copy(); data = None; resolution = wc.resolution
    n = len(cir); x = np.arange(n, dtype=float) if axis is None else np.asarray(axis, float); dx = float(x[1]-x[0])
    roi = np.ones(n, bool) if cfg.search_range is None else (x >= cfg.search_range[0]) & (x <= cfg.search_range[1])
    if frequency_context is not None:
        # The base extractor already validates the full complex input and context.
        data = _data(cir, frequency_context)
        if resolution is None:resolution = n*dx/(data["k"][-1]-data["k"][0]+1)
    if resolution is not None:extra["win_resolution"] = float(resolution)
    if data is not None and len(base):
        rect = np.ones(len(data["z"])); weights = [rect]+[_taper(name, data, wc) for name in wc.windows]
        views = [_view(data, w) for w in weights]; amplitudes = [abs(v) if v is not None else None for v in views]
        shapes = [_shape(a) if a is not None else None for a in amplitudes]
        original = amplitudes[0]; maximum = float(original[roi].max())
        floor = max(maximum*1e-12, np.finfo(float).tiny)
        positions = (base.position.to_numpy(float)-x[0])/dx
        fixed = [_evaluate(data, w, positions) if v is not None else np.full(len(base), np.nan+0j)
                 for w, v in zip(weights, views)]
        ranks = []
        for values in fixed:
            magnitude = abs(values); divisor = max(float(np.nanmax(magnitude)), floor)
            order = np.argsort(-np.round(magnitude/divisor, 12), kind="stable")
            rank = np.empty(len(order), int); rank[order] = np.arange(len(order)); ranks.append(rank)
        strong = sorted(range(len(shapes[0]["peaks"])), key=lambda j: (-round(float(original[shapes[0]["peaks"][j]]/max(maximum, floor)), 12), int(shapes[0]["peaks"][j]))) if wc.include_template_features else []
        references = []
        for j in strong:
            sample = float(shapes[0]["samples"][j])
            if original[shapes[0]["peaks"][j]] <= floor:continue
            if all(abs(_delta(sample, old, n))*dx >= wc.reference_min_separation_resolution*resolution for old in references):references.append(sample)
        for row_number, (index, sample) in enumerate(zip(base.index, positions)):
            left, right = _bounds(shapes[0]["valleys"], sample, n)
            extra.at[index, "win_valley_bounds_available"] = float(np.isfinite(left) and np.isfinite(right))
            extra.at[index, "win_left_valley_gap_resolution"] = -left*dx/resolution
            extra.at[index, "win_right_valley_gap_resolution"] = right*dx/resolution
            a0 = float(abs(fixed[0][row_number])); step = wc.derivative_step_resolution*resolution/dx
            baseline_side = abs(_evaluate(data, rect, [sample-step, sample+step]))
            slope0 = _ratio(baseline_side[1]-baseline_side[0], a0, floor)
            curvature0 = _ratio(2*a0-baseline_side.sum(), a0, floor)
            candidates0 = np.flatnonzero(abs(_delta(shapes[0]["samples"], sample, n)) <= max(.55, .01*resolution/dx))
            b0 = int(candidates0[np.argmin(abs(_delta(shapes[0]["samples"][candidates0], sample, n)))]) if len(candidates0) else None
            prominence0 = shapes[0]["prominence"][b0] if b0 is not None else np.nan
            width0 = shapes[0]["widths"][b0] if b0 is not None else np.nan
            supports, offsets, log_ratios = [], [], []
            for j, name in enumerate(wc.windows, 1):
                prefix = f"win_{name}_"; w = weights[j]
                if views[j] is None:continue
                a = amplitudes[j]; shape = shapes[j]; aw = float(abs(fixed[j][row_number])); mw = float(a[roi].max())
                side = abs(_evaluate(data, w, [sample-step, sample+step]))
                slope = _ratio(side[1]-side[0], aw, floor); curvature = _ratio(2*aw-side.sum(), aw, floor)
                amp_log = _log_ratio(aw, a0, floor); global_log = _log_ratio(mw, maximum, floor)
                values = dict(coherent_gain=float(w.mean()), fixed_amp_log_ratio=amp_log, global_max_log_ratio=global_log,
                    relative_height=_ratio(aw, mw, floor), relative_height_change=_ratio(aw, mw, floor)-_ratio(a0, maximum, floor),
                    relative_height_log_ratio=amp_log-global_log,
                    amplitude_rank_change=float((ranks[j][row_number]-ranks[0][row_number])/max(len(base)-1, 1)),
                    fixed_slope=slope, slope_change=slope-slope0, fixed_curvature=curvature, curvature_change=curvature-curvature0,
                    peak_support=0.)
                if aw > floor and a0 > floor:
                    phase = np.angle(fixed[j][row_number]*np.conj(fixed[0][row_number]))
                    values.update(phase_cosine=float(np.cos(phase)), phase_sine=float(np.sin(phase)))
                for key, delta in [("left_valley_fill", left), ("right_valley_fill", right)]:
                    if np.isfinite(delta):values[key] = _ratio(float(abs(_evaluate(data, w, [sample+delta])[0])), aw, floor)
                delta = _delta(shape["samples"], sample, n)
                valid = abs(delta)*dx <= wc.match_radius_resolution*resolution
                peak_coordinates = x[0]+shape["samples"]*dx
                if cfg.search_range is not None:valid &= (peak_coordinates >= cfg.search_range[0]) & (peak_coordinates <= cfg.search_range[1])
                if np.isfinite(left):valid &= delta > left
                if np.isfinite(right):valid &= delta < right
                ids = np.flatnonzero(valid)
                if len(ids):
                    match = int(ids[np.argmin(abs(delta[ids]))]); matched_sample = shape["samples"][match]
                    matched_amp = float(abs(_evaluate(data, w, [matched_sample])[0]))
                    values.update(peak_support=1., peak_offset_resolution=float(delta[match]*dx/resolution),
                        matched_amp_log_ratio=_log_ratio(matched_amp, a0, floor),
                        prominence_log_ratio=_log_ratio(shape["prominence"][match], prominence0, floor),
                        matched_width_resolution=float(shape["widths"][match]*dx/resolution),
                        matched_width_ratio=_ratio(shape["widths"][match], width0, 1e-12))
                    offsets.append(abs(values["peak_offset_resolution"]))
                wl, wr = _bounds(shape["valleys"], sample, n)
                if np.isfinite(wl) and np.isfinite(wr):
                    all_delta = _delta(positions, sample, n)
                    values["basin_candidate_count"] = int(np.sum((all_delta > wl) & (all_delta < wr)))
                supports.append(values["peak_support"])
                if np.isfinite(amp_log):log_ratios.append(amp_log)
                for key, value in values.items():extra.at[index, prefix+key] = value
            if supports:extra.at[index, "win_peak_support_fraction"] = float(np.mean(supports))
            if offsets:extra.at[index, "win_peak_offset_mad_resolution"] = float(np.median(offsets))
            if log_ratios:extra.at[index, "win_fixed_amp_log_ratio_std"] = float(np.std(log_ratios))
            if wc.include_template_features and a0 > floor and all(v is not None for v in views):
                target = np.array([v[row_number] for v in fixed]); self_error = _nmse(target, np.ones(len(target), complex))
                extra.at[index, "win_self_template_nmse"] = self_error
                eligible = [s for s in references if abs(_delta(s, sample, n))*dx > wc.candidate_exclusion_resolution*resolution]
                eligible.sort(key=lambda s: (-round(float(abs(_evaluate(data, rect, [s])[0])/max(maximum, floor)), 12), float(_delta(s, sample, n))))
                eligible = eligible[:wc.max_reference_peaks]
                best = None
                for ref in eligible:
                    # Templates represent unshifted zero-delay PSFs; convert only their relative delay.
                    relative = float(_delta(sample, ref, n))
                    template_sample = relative+(n//2 if data["shifted"] else 0)
                    template = np.array([_evaluate(data, w, [template_sample], template=True)[0] for w in weights])
                    error = _nmse(target, template)
                    if np.isfinite(error) and (best is None or error < best[0]-1e-12):best = (error, ref)
                if best is not None:
                    error, ref = best
                    extra.at[index, "win_sidelobe_template_nmse"] = error
                    extra.at[index, "win_sidelobe_minus_self_nmse"] = error-self_error
                    extra.at[index, "win_sidelobe_reference_gap_resolution"] = float(_delta(sample, ref, n)*dx/resolution)
                    extra.at[index, "win_sidelobe_reference_relative_height"] = _ratio(float(abs(_evaluate(data, rect, [ref])[0])), maximum, floor)
    extra.replace([np.inf, -np.inf], np.nan, inplace=True)
    result = pd.concat([base, extra], axis=1); result.attrs = attrs
    result.attrs.update(window_config=asdict(wc), window_feature_version="1.0",
        window_feature_availability=dict(rewindowing=data is not None, complex_phase=data is not None,
                                        template_comparison=data is not None and wc.include_template_features),
        window_notes=["Original candidate rows and base columns are unchanged; no extra amplitude rejection.",
            "Reference is rectangular reconstruction of recovered measured tones, even for a prewindowed input.",
            "Tapers use physical FFT-bin span; missing tones are not compressed or invented.",
            "Global maxima and amplitude ranks use the same search range/original candidate set.",
            "Peak matching is restricted to original rectangular valleys and a bandwidth-scaled radius.",
            "A vanished/merged peak is evidence only; genuine nearby components may merge too.",
            "Multiwindow self/side fits are local hypotheses, not proof of a physical path.",
            "NaN denotes unavailable, null-amplitude or undefined geometric features."])
    return result
