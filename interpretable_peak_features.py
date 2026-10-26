"""Interpretable, truth-independent BLE peak-candidate features.

Each row describes one candidate on one IQ path. The 512-point IFFT is an
interpolated response of 80 one-MHz tones, not a UWB impulse response.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths

NFFT = 512
PERIOD_M = 299792458 / (2e6)
GRID_M = np.fft.fftshift(np.fft.fftfreq(NFFT)) * PERIOD_M
TONE_INDEX = np.arange(80)


def candidate_features(z: np.ndarray, d0: np.ndarray, candidates: np.ndarray,
                       file_ids=None, path_ids=None, frame_ids=None) -> pd.DataFrame:
    """Extract candidate and global features; candidates have shape [N, K].

    Missing candidates are NaN. Input order for each file/path must be causal.
    The optional history features use at most the preceding five frames.
    """
    z = np.asarray(z)
    d0 = np.asarray(d0, float)
    candidates = np.asarray(candidates, float)
    if z.ndim != 2 or z.shape[1] != 80 or candidates.shape[0] != len(z):
        raise ValueError("Expected IQ [N,80] and candidates [N,K]")
    n = len(z)
    file_ids = np.zeros(n, int) if file_ids is None else np.asarray(file_ids)
    path_ids = np.zeros(n, int) if path_ids is None else np.asarray(path_ids)
    frame_ids = np.arange(n) if frame_ids is None else np.asarray(frame_ids)
    spectrum = np.abs(np.fft.fftshift(np.fft.ifft(z, n=NFFT, axis=1), axes=1))
    rows = []
    history: dict[tuple, deque] = {}
    for i in range(n):
        v = spectrum[i]
        power = v * v
        positive = (GRID_M >= 0) & (GRID_M <= 60)
        total = max(float(power[positive].sum()), 1e-20)
        main = max(float(v[positive].max()), 1e-20)
        p = power[positive] / total
        entropy = float(-(p * np.log(np.maximum(p, 1e-20))).sum() / np.log(len(p)))
        floor_mask = positive & (np.abs(GRID_M - d0[i]) > 3.0)
        floor = float(np.median(v[floor_mask])) if floor_mask.any() else 0.0
        floor_ratio = floor / main
        tone_amp = np.abs(z[i])
        amp_cv = float(tone_amp.std() / max(tone_amp.mean(), 1e-20))
        peaks, _ = find_peaks(v)
        positive_peaks = peaks[positive[peaks]]
        if len(positive_peaks):
            prominence = peak_prominences(v, positive_peaks)[0]
            widths = peak_widths(v, positive_peaks, rel_height=.5)[0] * PERIOD_M / NFFT
        else:
            prominence = widths = np.empty(0)
        key = (file_ids[i], path_ids[i])
        past = history.setdefault(key, deque(maxlen=5))
        frame_candidates = []
        for j, distance in enumerate(candidates[i]):
            if not np.isfinite(distance) or distance < 0 or distance > 60:
                continue
            bin_id = int(np.argmin(np.abs(GRID_M - distance)))
            amplitude = float(v[bin_id])
            if len(positive_peaks):
                nearest = int(np.argmin(np.abs(GRID_M[positive_peaks] - distance)))
                pk = int(positive_peaks[nearest])
                is_local_peak = abs(float(GRID_M[pk]) - distance) <= .5
                other = positive_peaks[positive_peaks != pk]
                competitor_gap = float(np.min(np.abs(GRID_M[other] - distance))) if len(other) else np.nan
                local_count = int(np.sum(np.abs(GRID_M[positive_peaks] - distance) <= 5))
            else:
                nearest = 0
                is_local_peak = False
                competitor_gap = np.nan
                local_count = 0
            pre = (GRID_M >= distance - 3) & (GRID_M < distance - .5)
            post = (GRID_M > distance + .5) & (GRID_M <= distance + 3)
            near = np.abs(GRID_M - distance) <= 1
            pre_energy = float(power[pre].sum())
            post_energy = float(power[post].sum())
            early = (GRID_M >= 0) & (GRID_M < distance - 1)
            # A candidate delay derotates its expected linear phase across tones.
            rotated = z[i] * np.exp(2j * np.pi * TONE_INDEX * distance / PERIOD_M)
            phase_coherence = float(abs(rotated.sum()) / max(np.abs(rotated).sum(), 1e-20))
            blocks = rotated.reshape(4, 20).sum(axis=1)
            subband_coherence = float(abs(blocks.sum()) / max(np.abs(blocks).sum(), 1e-20))
            matches = []
            for old in past:
                if len(old):
                    gap = np.abs(old[:, 0] - distance)
                    k = int(gap.argmin())
                    if gap[k] <= 1.0:
                        matches.append(old[k])
            matched_dist = np.asarray([m[0] for m in matches], float)
            matched_amp = np.asarray([m[1] for m in matches], float)
            rows.append(dict(file=file_ids[i], path=int(path_ids[i]), frame=frame_ids[i],
                candidate_rank=j, candidate_m=float(distance),
                gap_from_ifft_m=float(d0[i] - distance),
                earlier_than_ifft=bool(distance <= d0[i] - .3),
                relative_gap=float((d0[i] - distance) / (d0[i] + 5)),
                relative_height=amplitude / main,
                candidate_to_floor=amplitude / max(floor, 1e-20),
                prominence_ratio=float(prominence[nearest] / main) if is_local_peak else np.nan,
                half_prominence_width_m=float(widths[nearest]) if is_local_peak else np.nan,
                nearest_competing_peak_m=competitor_gap,
                local_peak_count_5m=local_count,
                energy_within_1m=float(power[near].sum() / total),
                energy_before_3m=pre_energy / total,
                energy_after_3m=post_energy / total,
                rise_asymmetry=(post_energy - pre_energy) / max(post_energy + pre_energy, 1e-20),
                earlier_energy=float(power[early].sum() / total),
                phase_coherence=phase_coherence,
                subband_coherence=subband_coherence,
                spectrum_entropy=entropy,
                spectrum_floor_ratio=floor_ratio,
                tone_amplitude_cv=amp_cv,
                history_support=len(matches),
                history_jitter_m=float(np.median(abs(matched_dist - distance))) if len(matches) else np.nan,
                history_height_ratio=float(amplitude / np.median(matched_amp)) if len(matches) else np.nan))
            frame_candidates.append((distance, amplitude))
        past.append(np.asarray(frame_candidates, float).reshape(-1, 2))
    return pd.DataFrame(rows)
