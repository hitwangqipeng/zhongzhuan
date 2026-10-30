"""BLE 频域 IQ / IFFT CIR 的数值核心（仅依赖 NumPy）。

每个元素对应一个频率，所有频点始终留在原始均匀网格上。
用户可直接修改 ``amplitude_scale`` 和 ``phase_delta_deg``，或调用
``set_point`` / ``set_range``；它们都是相对于理想 IQ 的总倍率/总偏移：

    IQ[k] = ideal_IQ[k] * amplitude_scale[k] * exp(j * phase_delta_deg[k])

相位偏移的单位是度。倍率为 0 表示缺失频点，不能把该频点从数组删除。
80 个 1 MHz 间隔的默认频点是合成频网，不代表标准 BLE CS 有效信道表。
近邻多径预设直接构造最终输入 IQ 的两条等效时延分量，没有模拟单程
信道 H(f) 到双方 PCT 乘积 H(f)^2 的全部交叉项。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path

import numpy as np


PRESETS = {
    "ideal": "理想单径",
    "edge_gaussian": "两端幅度下降",
    "edge_missing": "边缘频点缺失",
    "hann": "Hann 幅度加权",
    "hamming": "Hamming 幅度加权",
    "quadratic_phase": "二次相位误差",
    "linear_phase": "纯线性相位",
    "close_multipath": "近邻多径",
    "center_missing": "中间频点缺失",
}


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是有限实数。")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} 必须是有限实数。")
    return value


def _integer(value: Integral, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是整数。")
    return int(value)


@dataclass
class Config:
    n_tones: int = 80
    df_hz: float = 1e6
    fc_hz: float = 2.44e9
    nfft: int = 16384
    distance_m: float = 8.0
    round_trip: bool = True
    phase0_rad: float = 0.4
    c: float = 299792458.0
    db_floor: float = -60.0

    def validate(self) -> None:
        n = _integer(self.n_tones, "频点数")
        nfft = _integer(self.nfft, "IFFT 点数")
        if n < 2:
            raise ValueError("频点数至少为 2。")
        if nfft < n or nfft % 2:
            raise ValueError("IFFT 点数必须是大于等于频点数的偶数。")
        for name in ("df_hz", "fc_hz", "c"):
            if _finite_real(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} 必须大于 0。")
        for name in ("distance_m", "phase0_rad", "db_floor"):
            _finite_real(getattr(self, name), name)
        if self.db_floor >= 0:
            raise ValueError("显示底限 db_floor 必须小于 0 dB。")
        if not isinstance(self.round_trip, (bool, np.bool_)):
            raise ValueError("round_trip 必须是 True 或 False。")


@dataclass
class Analysis:
    tau_s: np.ndarray
    distance_m: np.ndarray
    cir: np.ndarray
    amplitude: np.ndarray
    normalized_amplitude: np.ndarray
    db: np.ndarray
    phase_unwrapped_rad: np.ndarray
    phase_residual_rad: np.ndarray
    peak_distance_m: float
    width_3db_m: float
    width_3db_ns: float
    peak_amplitude: float
    left_crossing_m: float
    right_crossing_m: float


class IQModel:
    """可逐频点修改的理想 IQ 及其 IFFT 分析。

    频点下标从 0 开始；set_range 的 start、stop 两端都包含。
    预设会替换当前全部幅相设置；之后的单点修改是覆盖而非累加。
    配置在初始化后视为固定；改频点数等结构参数时应创建新模型。
    """

    def __init__(self, config: Config | None = None):
        self.cfg = replace(config) if config is not None else Config()
        self.cfg.validate()
        self.indices = np.arange(self.cfg.n_tones)
        self.f_offset_hz = (
            self.indices - (self.cfg.n_tones - 1) / 2
        ) * self.cfg.df_hz
        self.frequencies_hz = self.cfg.fc_hz + self.f_offset_hz
        tau0 = self.propagation_factor * self.cfg.distance_m / self.cfg.c
        self._ideal_phase = self.cfg.phase0_rad - 2 * np.pi * self.f_offset_hz * tau0
        self.ideal_iq = np.exp(1j * self._ideal_phase)
        self.amplitude_scale = np.ones(self.cfg.n_tones)
        self.phase_delta_deg = np.zeros(self.cfg.n_tones)
        self.preset_name = "ideal"
        self.true_path_distances_m = np.array([self.cfg.distance_m])

    @property
    def propagation_factor(self) -> int:
        return 2 if self.cfg.round_trip else 1

    def _validate_controls(self) -> tuple[np.ndarray, np.ndarray]:
        amp = np.asarray(self.amplitude_scale, dtype=float)
        phase = np.asarray(self.phase_delta_deg, dtype=float)
        expected = (self.cfg.n_tones,)
        if amp.shape != expected or phase.shape != expected:
            raise ValueError(f"幅度和相位数组都必须是长度 {self.cfg.n_tones} 的一维数组。")
        if not np.all(np.isfinite(amp)) or np.any(amp < 0):
            raise ValueError("所有幅度倍率必须是大于等于 0 的有限实数。")
        if not np.all(np.isfinite(phase)):
            raise ValueError("所有相位偏移必须是有限实数，单位为度。")
        return amp, phase

    @property
    def iq(self) -> np.ndarray:
        amp, phase = self._validate_controls()
        # 保存用户原始角度供显示；计算前取模，360 度的整数倍不改变 IQ。
        return self.ideal_iq * amp * np.exp(1j * np.deg2rad(np.remainder(phase, 360)))

    def _index(self, index: int) -> int:
        index = _integer(index, "频点下标")
        if not 0 <= index < self.cfg.n_tones:
            raise ValueError(f"频点下标必须在 0 到 {self.cfg.n_tones-1} 之间。")
        return index

    @staticmethod
    def _edit_values(amplitude_scale, phase_delta_deg):
        amplitude = phase = None
        if amplitude_scale is not None:
            amplitude = _finite_real(amplitude_scale, "幅度倍率")
            if amplitude < 0:
                raise ValueError("幅度倍率必须大于等于 0。")
        if phase_delta_deg is not None:
            phase = _finite_real(phase_delta_deg, "相位偏移")
        return amplitude, phase

    def set_point(self, index: int, amplitude_scale=None, phase_delta_deg=None) -> None:
        index = self._index(index)
        amplitude, phase = self._edit_values(amplitude_scale, phase_delta_deg)
        self._validate_controls()
        if amplitude is not None:
            self.amplitude_scale[index] = amplitude
        if phase is not None:
            self.phase_delta_deg[index] = phase
        if amplitude is not None or phase is not None:
            self._mark_custom()

    def set_range(self, start: int, stop: int,
                  amplitude_scale=None, phase_delta_deg=None) -> None:
        start, stop = self._index(start), self._index(stop)
        if stop < start:
            raise ValueError("结束下标必须大于等于起始下标。")
        amplitude, phase = self._edit_values(amplitude_scale, phase_delta_deg)
        self._validate_controls()
        if amplitude is not None:
            self.amplitude_scale[start:stop+1] = amplitude
        if phase is not None:
            self.phase_delta_deg[start:stop+1] = phase
        if amplitude is not None or phase is not None:
            self._mark_custom()

    def _mark_custom(self):
        self.preset_name = "custom"
        self.true_path_distances_m = np.array([], dtype=float)

    def apply_preset(self, key: str) -> None:
        if key not in PRESETS:
            raise ValueError(f"未知预设：{key!r}。可用预设：{', '.join(PRESETS)}")
        n = self.cfg.n_tones
        bandwidth = n * self.cfg.df_hz
        f = self.f_offset_hz
        amp = np.ones(n)
        phase = np.zeros(n)
        paths = np.array([self.cfg.distance_m])
        if key == "edge_gaussian":
            amp = np.exp(-0.5 * (f / (0.23 * bandwidth))**2)
        elif key == "edge_missing":
            count = min(n // 2 - 1, max(1, int(round(n * 0.2))))
            if count:
                amp[:count] = amp[-count:] = 0.0
        elif key == "hann":
            amp = 0.5 - 0.5 * np.cos(2 * np.pi * self.indices / (n-1))
        elif key == "hamming":
            amp = 0.54 - 0.46 * np.cos(2 * np.pi * self.indices / (n-1))
        elif key == "quadratic_phase":
            phase = np.rad2deg(2.2 * (2 * f / bandwidth)**2)
        elif key == "linear_phase":
            dt = self.propagation_factor * 1.2 / self.cfg.c
            phase = np.rad2deg(-2 * np.pi * f * dt)
            paths = np.array([self.cfg.distance_m + 1.2])
        elif key == "close_multipath":
            dt = 0.7 / bandwidth
            ratio = 0.5 + 0.5 * np.exp(-2j * np.pi * f * dt)
            amp = np.abs(ratio)
            phase = np.rad2deg(np.unwrap(np.angle(ratio)))
            paths = np.array([
                self.cfg.distance_m,
                self.cfg.distance_m + self.cfg.c * dt / self.propagation_factor,
            ])
        elif key == "center_missing":
            count = min(n-1, max(1, int(round(n * 0.125))))
            start = (n-count) // 2
            amp[start:start+count] = 0.0
        self.amplitude_scale = amp
        self.phase_delta_deg = phase
        self.preset_name = key
        self.true_path_distances_m = paths

    def analyze(self, iq=None) -> Analysis:
        z = self.iq if iq is None else np.asarray(iq, dtype=complex)
        if z.shape != (self.cfg.n_tones,):
            raise ValueError(f"IQ 必须是长度 {self.cfg.n_tones} 的一维复数数组。")
        if not np.all(np.isfinite(z)):
            raise ValueError("IQ 不得含有 NaN 或无穷值。")
        cfg = self.cfg
        tau = (np.arange(cfg.nfft) - cfg.nfft // 2) / (cfg.nfft * cfg.df_hz)
        distance = cfg.c * tau / self.propagation_factor
        # 递增排列的频点直接末尾补零；不要先 ifftshift 那 N 个原始频点。
        input_scale = float(np.max(np.abs(z)))
        if input_scale:
            normalized_cir = np.fft.fftshift(np.fft.ifft(z/input_scale, cfg.nfft))
            normalized_cir *= cfg.nfft / cfg.n_tones
            normalized_cir *= np.exp(2j * np.pi * self.f_offset_hz[0] * tau)
            # 连续频率定义：sum(z[k]*exp(j*2*pi*f_offset[k]*tau))/N。
            cir = normalized_cir * input_scale
        else:
            cir = np.zeros(cfg.nfft, dtype=complex)
        amplitude = np.abs(cir)
        peak = float(np.max(amplitude))
        if peak:
            normalized_amplitude = amplitude/peak
        else:
            normalized_amplitude = np.zeros(cfg.nfft)
        db = 20 * np.log10(np.maximum(normalized_amplitude, 10**(cfg.db_floor/20)))
        phase, phase_residual = self._phase_diagnostics(z)
        peak_distance = left_m = right_m = width_m = width_ns = np.nan
        # 单频或全零数据没有一个可定义的局部主峰，避免浮点噪声造假峰宽。
        if peak and np.count_nonzero(np.abs(z) > input_scale*1e-12) >= 2:
            p = int(np.argmax(normalized_amplitude))
            peak_distance = float(distance[p])
            if 0 < p < cfg.nfft-1:
                den = normalized_amplitude[p-1]-2*normalized_amplitude[p]+normalized_amplitude[p+1]
                if abs(den) > np.finfo(float).eps:
                    offset = np.clip(0.5*(normalized_amplitude[p-1]-normalized_amplitude[p+1])/den, -0.5, 0.5)
                    peak_distance += float(offset * (distance[1]-distance[0]))
            threshold = 1/np.sqrt(2)
            left = right = p
            while left > 0 and normalized_amplitude[left-1] >= threshold:
                left -= 1
            while right < cfg.nfft-1 and normalized_amplitude[right+1] >= threshold:
                right += 1
            if left > 0 and right < cfg.nfft-1:
                left_m = self._crossing(distance[left-1], distance[left], normalized_amplitude[left-1], normalized_amplitude[left], threshold)
                right_m = self._crossing(distance[right], distance[right+1], normalized_amplitude[right], normalized_amplitude[right+1], threshold)
                width_m = right_m-left_m
                width_ns = width_m*self.propagation_factor/cfg.c*1e9
        return Analysis(
            tau_s=tau, distance_m=distance, cir=cir, amplitude=amplitude,
            normalized_amplitude=normalized_amplitude, db=db,
            phase_unwrapped_rad=phase, phase_residual_rad=phase_residual,
            peak_distance_m=peak_distance, width_3db_m=width_m,
            width_3db_ns=width_ns, peak_amplitude=peak,
            left_crossing_m=left_m, right_crossing_m=right_m,
        )

    @staticmethod
    def _crossing(x1, x2, y1, y2, threshold) -> float:
        return float(x1 + (threshold-y1)*(x2-x1)/(y2-y1))

    def _phase_diagnostics(self, z):
        phase = np.full(self.cfg.n_tones, np.nan)
        residual = phase.copy()
        magnitude = np.abs(z)
        peak = float(np.max(magnitude))
        if not peak:
            return phase, residual
        valid = magnitude > peak*1e-6
        # 使用已知仿真参考，先移除大斜率理想相位，降低缺口处 unwrap 歧义。
        # 任意手改点可超过 +/-180 度；此图仅给出一个连续相位分支，编辑器
        # 应同时显示原始 phase_delta_deg，不能把 unwrap 分支误作用户输入。
        delta = np.unwrap(np.angle(z[valid] * np.conj(self.ideal_iq[valid])))
        phase[valid] = self._ideal_phase[valid] + delta
        if np.count_nonzero(valid) >= 2:
            f_mhz = self.f_offset_hz / 1e6
            fit = np.polyfit(f_mhz[valid], delta, 1)
            residual[valid] = delta - np.polyval(fit, f_mhz[valid])
        return phase, residual

    def save_csv(self, path: str | Path) -> Path:
        """保存完整频点编辑表；frequency_mhz 为绝对频率。"""
        amplitude, phase = self._validate_controls()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tone_index", "frequency_mhz", "amplitude_scale", "phase_delta_deg"])
            for k in self.indices:
                writer.writerow([
                    int(k), format(self.frequencies_hz[k]/1e6, ".17g"),
                    format(amplitude[k], ".17g"), format(phase[k], ".17g"),
                ])
        return target

    def load_csv(self, path: str | Path) -> None:
        """完整验证后一次性更新。无效 CSV 不会改变当前设置。"""
        required = {"tone_index", "frequency_mhz", "amplitude_scale", "phase_delta_deg"}
        amplitude = np.empty(self.cfg.n_tones)
        phase = np.empty(self.cfg.n_tones)
        seen = set()
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("CSV 必须包含 tone_index、frequency_mhz、amplitude_scale、phase_delta_deg 列。")
            for line_number, row in enumerate(reader, start=2):
                try:
                    index = self._index(int(row["tone_index"].strip()))
                    if index in seen:
                        raise ValueError(f"频点 {index} 重复。")
                    frequency = _finite_real(float(row["frequency_mhz"]), "频率")*1e6
                    if abs(frequency-self.frequencies_hz[index]) > 1.0:
                        raise ValueError(f"频点 {index} 的频率不符合当前网格。")
                    amp_value, phase_value = self._edit_values(
                        float(row["amplitude_scale"]), float(row["phase_delta_deg"])
                    )
                except (TypeError, AttributeError, ValueError, KeyError) as exc:
                    raise ValueError(f"CSV 第 {line_number} 行无效：{exc}") from exc
                amplitude[index] = amp_value
                phase[index] = phase_value
                seen.add(index)
        if len(seen) != self.cfg.n_tones:
            missing = sorted(set(self.indices)-seen)
            raise ValueError(f"CSV 必须完整包含 {self.cfg.n_tones} 个频点；缺少：{missing}")
        self.amplitude_scale = amplitude
        self.phase_delta_deg = phase
        self._mark_custom()


__all__ = ["Config", "Analysis", "IQModel", "PRESETS"]
