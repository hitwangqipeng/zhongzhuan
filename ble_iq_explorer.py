"""BLE 频域 IQ / CIR 交互实验：可逐频点改幅度倍率与相位偏移。

运行：python ble_iq_explorer.py
无窗口导出：python ble_iq_explorer.py --no-gui
仅依赖 NumPy、Matplotlib 和同目录的 ble_iq_model.py。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib

# 必须在导入 pyplot 之前选择无窗口后端。
if "--no-gui" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

from ble_iq_model import Config, IQModel, PRESETS


# ===================== 可以直接修改的配置区 =====================
CONFIG = Config(
    n_tones=80,
    df_hz=1e6,
    fc_hz=2.44e9,
    nfft=16384,
    distance_m=8.0,
    round_trip=True,
)
INITIAL_PRESET = "ideal"

# 频点索引从 0 开始；80 个点的合法索引为 0～79。
# 幅度是相对理想 IQ 的总倍率；相位是相对理想 IQ 的总偏移，单位为度。
# 例如取消下面两行注释，就可只修改第 10、25 号频点：
POINT_EDITS: dict[int, dict[str, float]] = {
    # 10: {"amplitude_scale": 0.5, "phase_delta_deg": 45.0},
    # 25: {"amplitude_scale": 1.2, "phase_delta_deg": -30.0},
}
# 最终 IQ = 理想 IQ × amplitude_scale × exp(1j × 相位偏移弧度)。
# 在预设基础上修改一个频点时，以上数值覆盖该点的“总倍率/总偏移”。
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "ble_iq_python"
# ===============================================================


def configure_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei",
                 "Noto Sans CJK SC", "PingFang SC"]
    selected = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [selected, "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b9c5d3", "axes.labelcolor": "#334155",
        "xtick.color": "#64748b", "ytick.color": "#64748b",
        "text.color": "#172033", "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _metrics(analysis) -> dict:
    return {name: _finite(getattr(analysis, name)) for name in (
        "peak_distance_m", "width_3db_m", "width_3db_ns", "peak_amplitude",
        "left_crossing_m", "right_crossing_m")}


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def _new_output_folder(root: Path, prefix: str = "snapshot") -> Path:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    folder.mkdir()
    return folder


class IQExplorer:
    """Matplotlib 交互界面，也可在 Agg 后端下直接测试编辑方法。"""

    IDEAL_COLOR = "#8593a6"
    CURRENT_COLOR = "#2563eb"

    def __init__(self, model: IQModel | None = None, output_dir: Path | str | None = None):
        configure_fonts()
        self.model = model if model is not None else IQModel(CONFIG)
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
        self.selected_index = 0
        self._syncing = False
        self._message = "拖动滑块，或输入精确值后回车。"
        self.reference = self.model.analyze(self.model.ideal_iq)
        self.current = self.model.analyze()
        self.fig = plt.figure(figsize=(17.5, 10.5), dpi=100)
        try:
            self.fig.canvas.manager.set_window_title("BLE IQ / CIR 逐频点交互实验")
        except AttributeError:
            pass
        grid = self.fig.add_gridspec(3, 2, left=0.055, right=0.715,
                                    top=0.918, bottom=0.073, hspace=0.48, wspace=0.25)
        self.axes = [self.fig.add_subplot(grid[row, col])
                     for row in range(3) for col in range(2)]
        self.fig.text(0.045, 0.968, "BLE 频域 IQ 与 CIR：逐频点修改", fontsize=19, weight="bold")
        self.subtitle = self.fig.text(0.045, 0.940, "", fontsize=10, color="#64748b")
        self.fig.text(0.045, 0.024,
                      "合成实验 · 灰虚线为理想 IQ · 幅度倍率/相位偏移可独立修改 · "
                      "CIR 各自按峰值归一化 · 宽度采用功率半高宽（幅度 0.707）",
                      fontsize=9, color="#64748b")
        self._build_controls()
        self.fig.canvas.mpl_connect("button_press_event", self._click_select)
        self._sync_controls()
        self.refresh()

    def _control_axes(self, y: float, height: float = 0.027, x: float = 0.79,
                      width: float = 0.18):
        return self.fig.add_axes([x, y, width, height], facecolor="#f1f5f9")

    def _build_controls(self) -> None:
        self.fig.text(0.775, 0.948, "修改频点", fontsize=14, weight="bold")
        self.fig.text(0.775, 0.925, "相位偏移：相对理想 IQ，单位为度", fontsize=9)
        self.preset_keys = list(PRESETS)
        self.preset_labels = [PRESETS[key] for key in self.preset_keys]
        active = self.preset_keys.index(self.model.preset_name) if self.model.preset_name in self.preset_keys else 0
        self.preset_radio = RadioButtons(self._control_axes(0.700, 0.213, 0.775, 0.205),
                                         self.preset_labels, active=active)
        for text in self.preset_radio.labels:
            text.set_fontsize(9)
        self.preset_radio.on_clicked(self._on_preset)
        self.fig.text(0.775, 0.681, "切换预设会替换当前全部频点修改。", fontsize=8, color="#64748b")

        self.point_label = self.fig.text(0.775, 0.654, "", fontsize=10, weight="bold")
        self.index_slider = Slider(self._control_axes(0.625), "", 0,
                                   self.model.cfg.n_tones - 1, valinit=0, valstep=1, valfmt="%d")
        self.index_slider.on_changed(lambda value: self._guard(self.select_point, int(value)))
        self.index_box = TextBox(self._control_axes(0.589, x=0.835, width=0.135), "索引 ", initial="0")
        self.index_box.on_submit(self._on_index_text)

        self.fig.text(0.775, 0.556, "幅度倍率（0 表示该频点置零）", fontsize=9)
        self.amplitude_slider = Slider(self._control_axes(0.527), "", 0.0, 3.0,
                                       valinit=1.0, valfmt="%.3f")
        self.amplitude_slider.on_changed(lambda value: self._guard(self.set_selected, amplitude_scale=float(value)))
        self.amplitude_box = TextBox(self._control_axes(0.491, x=0.835, width=0.135), "精确值 ", initial="1")
        self.amplitude_box.on_submit(lambda text: self._guard(
            lambda: self.set_selected(amplitude_scale=float(text))))

        self.fig.text(0.775, 0.458, "相位偏移 / 度", fontsize=9)
        self.phase_slider = Slider(self._control_axes(0.429), "", -180.0, 180.0,
                                   valinit=0.0, valfmt="%.1f°")
        self.phase_slider.on_changed(lambda value: self._guard(self.set_selected, phase_delta_deg=float(value)))
        self.phase_box = TextBox(self._control_axes(0.393, x=0.835, width=0.135), "精确值 ", initial="0")
        self.phase_box.on_submit(lambda text: self._guard(
            lambda: self.set_selected(phase_delta_deg=float(text))))

        self.fig.text(0.775, 0.362, "范围修改：使用当前点的倍率与偏移", fontsize=9)
        self.range_start_box = TextBox(self._control_axes(0.329, x=0.813, width=0.050), "从 ", initial="0")
        self.range_stop_box = TextBox(self._control_axes(0.329, x=0.913, width=0.050), "到 ",
                                      initial=str(self.model.cfg.n_tones - 1))
        self.range_button = Button(self._control_axes(0.291, 0.027, 0.79, 0.18), "应用到范围（含两端）")
        self.range_button.on_clicked(self._on_range)
        self.reset_point_button = Button(self._control_axes(0.249, 0.028, 0.775, 0.098), "重置当前点")
        self.reset_all_button = Button(self._control_axes(0.249, 0.028, 0.883, 0.098), "重置全部")
        self.reset_point_button.on_clicked(lambda _: self._guard(self.set_selected, amplitude_scale=1.0, phase_delta_deg=0.0))
        self.reset_all_button.on_clicked(lambda _: self._guard(self.apply_preset, "ideal"))

        self.csv_path_box = TextBox(self._control_axes(0.201, 0.028, 0.813, 0.168), "CSV ",
                                    initial="iq_edits.csv")
        self.save_csv_button = Button(self._control_axes(0.158, 0.028, 0.775, 0.098), "保存 CSV")
        self.load_csv_button = Button(self._control_axes(0.158, 0.028, 0.883, 0.098), "读取 CSV")
        self.save_csv_button.on_clicked(lambda _: self._guard(self._save_csv))
        self.load_csv_button.on_clicked(lambda _: self._guard(self._load_csv))
        self.snapshot_button = Button(self._control_axes(0.116, 0.028, 0.79, 0.18), "保存图像与完整数据")
        self.snapshot_button.on_clicked(lambda _: self._guard(self._save_snapshot_clicked))
        self.status_text = self.fig.text(0.775, 0.092, "", va="top", fontsize=8, color="#334155",
                                         linespacing=1.5)

    def _guard(self, function, *args, **kwargs):
        if self._syncing:
            return
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError, OSError, IndexError) as error:
            self._message = "输入/文件错误：" + str(error)
            self._update_status()
            self.fig.canvas.draw_idle()

    @staticmethod
    def _parse_index(text: str) -> int:
        value = float(text)
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError("频点索引必须是整数。")
        return int(value)

    def _on_index_text(self, text: str) -> None:
        if not self._syncing:
            self._guard(lambda: self.select_point(self._parse_index(text)))

    def _on_preset(self, label: str) -> None:
        if not self._syncing:
            self._guard(self.apply_preset, self.preset_keys[self.preset_labels.index(label)])

    def _on_range(self, _) -> None:
        def run():
            self.apply_range(self._parse_index(self.range_start_box.text),
                             self._parse_index(self.range_stop_box.text),
                             amplitude_scale=float(self.model.amplitude_scale[self.selected_index]),
                             phase_delta_deg=float(self.model.phase_delta_deg[self.selected_index]))
        self._guard(run)

    def _click_select(self, event) -> None:
        if event.inaxes not in (self.axes[1], self.axes[2]) or event.xdata is None:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        index = int(np.argmin(abs(self.model.f_offset_hz / 1e6 - event.xdata)))
        self._guard(self.select_point, index)

    def _sync_controls(self) -> None:
        self._syncing = True
        widgets = (self.index_slider, self.amplitude_slider, self.phase_slider,
                   self.index_box, self.amplitude_box, self.phase_box, self.preset_radio)
        previous_drawon = [widget.drawon for widget in widgets]
        for widget in widgets:
            widget.drawon = False
        try:
            index = self.selected_index
            amp = float(self.model.amplitude_scale[index])
            phase = float(self.model.phase_delta_deg[index])
            if amp > self.amplitude_slider.valmax:
                upper = float(max(3.0, np.ceil(amp * 1.1)))
                self.amplitude_slider.valmax = upper
                self.amplitude_slider.ax.set_xlim(0, upper)
            if abs(phase) > self.phase_slider.valmax:
                upper = float(max(180.0, np.ceil(abs(phase) / 180.0) * 180.0))
                self.phase_slider.valmin = -upper
                self.phase_slider.valmax = upper
                self.phase_slider.ax.set_xlim(-upper, upper)
            self.index_slider.set_val(index)
            self.amplitude_slider.set_val(amp)
            self.phase_slider.set_val(phase)
            self.index_box.set_val(str(index))
            self.amplitude_box.set_val(f"{amp:.8g}")
            self.phase_box.set_val(f"{phase:.8g}")
            if self.model.preset_name in self.preset_keys:
                active = self.preset_keys.index(self.model.preset_name)
                self.preset_radio.set_active(active)
            self.point_label.set_text(
                f"频点 {index} / {self.model.cfg.n_tones-1}   "
                f"{self.model.frequencies_hz[index]/1e6:.1f} MHz")
        finally:
            for widget, drawon in zip(widgets, previous_drawon):
                widget.drawon = drawon
            self._syncing = False

    def select_point(self, index: int) -> None:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError("频点索引必须是整数。")
        if not 0 <= index < self.model.cfg.n_tones:
            raise ValueError(f"频点索引范围为 0～{self.model.cfg.n_tones-1}。")
        self.selected_index = int(index)
        self._sync_controls()
        self.refresh()

    def set_selected(self, amplitude_scale=None, phase_delta_deg=None) -> None:
        self.model.set_point(self.selected_index, amplitude_scale=amplitude_scale,
                             phase_delta_deg=phase_delta_deg)
        self._message = f"已更新频点 {self.selected_index}。"
        self._sync_controls()
        self.refresh()

    def apply_preset(self, key: str) -> None:
        self.model.apply_preset(key)
        self._message = "已应用预设：" + PRESETS[key]
        self._sync_controls()
        self.refresh()

    def apply_range(self, start: int, stop: int, amplitude_scale=None,
                    phase_delta_deg=None) -> None:
        self.model.set_range(start, stop, amplitude_scale=amplitude_scale,
                             phase_delta_deg=phase_delta_deg)
        self._message = f"已更新频点 {start}～{stop}（含两端）。"
        self._sync_controls()
        self.refresh()

    def _save_csv(self) -> None:
        path = self._csv_file_path()
        self.model.save_csv(path)
        self._message = "已保存 CSV：" + path.name
        self._update_status()
        self.fig.canvas.draw_idle()

    def _load_csv(self) -> None:
        path = self._csv_file_path()
        self.model.load_csv(path)
        self._message = "已读取 CSV：" + path.name
        self._sync_controls()
        self.refresh()

    def _csv_file_path(self) -> Path:
        path = Path(self.csv_path_box.text.strip()).expanduser()
        if not self.csv_path_box.text.strip():
            raise ValueError("请输入 CSV 文件名或路径。")
        return path if path.is_absolute() else self.output_dir / path

    def _update_status(self) -> None:
        current = self.current
        ratio = current.peak_amplitude / self.reference.peak_amplitude
        message = self._message
        if len(message) > 28:
            message = message[:27] + "…"
        self.status_text.set_text(
            f"峰位 {_fmt(current.peak_distance_m)} m   宽度 {_fmt(current.width_3db_m)} m\n"
            f"实际峰高 / 理想峰高：{_fmt(ratio)}\n{message}")

    def refresh(self) -> None:
        self.current = self.model.analyze()
        f = self.model.f_offset_hz / 1e6
        z0 = self.model.ideal_iq
        z = self.model.iq
        current, ref = self.current, self.reference
        index = self.selected_index
        for ax in self.axes:
            ax.clear()
            ax.grid(axis="both", color="#e2e8f0", linewidth=0.65)
        a = self.axes
        a[0].plot(f, z0.real, "--", color="#94a3b8", lw=1.2, label="理想 I")
        a[0].plot(f, z0.imag, ":", color="#7ba69c", lw=1.4, label="理想 Q")
        a[0].plot(f, z.real, color="#2563eb", lw=1.3, label="当前 I")
        a[0].plot(f, z.imag, color="#c2410c", lw=1.3, label="当前 Q")
        a[0].set(title="频域 I / Q 分量", xlabel="相对中心频率 / MHz", ylabel="I、Q")
        a[0].legend(loc="upper right", ncol=2, fontsize=8)

        a[1].plot(f, abs(z0), "--", color=self.IDEAL_COLOR, lw=1.5, label="理想")
        a[1].plot(f, abs(z), ".-", color=self.CURRENT_COLOR, lw=1.4, ms=3, label="当前")
        a[1].plot(f[index], abs(z[index]), "o", color="#e11d48", ms=7)
        a[1].set(title="IQ 幅度：可点击曲线选择频点", xlabel="相对中心频率 / MHz", ylabel="幅度（统一尺度）")
        a[1].set_ylim(0, max(1.1, float(np.max(abs(z))) * 1.12))
        a[1].legend(loc="upper right", fontsize=8)

        a[2].axhline(0, ls="--", color=self.IDEAL_COLOR, lw=1.5, label="理想：0°")
        a[2].plot(f, self.model.phase_delta_deg, ".-", color="#c2410c", lw=1.3, ms=3,
                  label="配置的总相位偏移")
        a[2].plot(f[index], self.model.phase_delta_deg[index], "o", color="#e11d48", ms=7)
        a[2].set(title="相对理想 IQ 的相位偏移（可点击选点）", xlabel="相对中心频率 / MHz", ylabel="相位偏移 / 度")
        a[2].legend(loc="upper right", fontsize=8)

        a[3].plot(f, ref.phase_residual_rad, "--", color=self.IDEAL_COLOR, lw=1.5, label="理想")
        a[3].plot(f, current.phase_residual_rad, color="#7c3aed", lw=1.4, label="当前")
        a[3].set(title="各自去除最佳线性相位后的残差", xlabel="相对中心频率 / MHz", ylabel="相位残差 / rad")
        a[3].legend(loc="upper right", fontsize=8)
        finite_residual = current.phase_residual_rad[np.isfinite(current.phase_residual_rad)]
        if not finite_residual.size or np.max(abs(finite_residual)) < 1e-6:
            # 不把浮点舍入造成的 1e-17 rad 误差放大为看似显著的相位起伏。
            a[3].set_ylim(-0.05, 0.05)

        for ax in (a[4], a[5]):
            ax.plot(ref.distance_m, ref.db if ax is a[4] else ref.normalized_amplitude,
                    "--", color=self.IDEAL_COLOR, lw=1.6,
                    label=f"理想宽度 {_fmt(ref.width_3db_m)} m")
            ax.plot(current.distance_m, current.db if ax is a[4] else current.normalized_amplitude,
                    color=self.CURRENT_COLOR, lw=1.6,
                    label=f"当前宽度 {_fmt(current.width_3db_m)} m")
            ax.legend(loc="upper right", fontsize=8)
        distance_label = "双程等效距离 / m" if self.model.cfg.round_trip else "单程距离 / m"
        a[4].set(title="IFFT CIR（各自按峰值归一化）", xlabel=distance_label, ylabel="归一化幅度 / dB")
        a[4].set_xlim(max(float(ref.distance_m[0]), self.model.cfg.distance_m - 8),
                      min(float(ref.distance_m[-1]), self.model.cfg.distance_m + 10))
        a[4].set_ylim(self.model.cfg.db_floor, 1)
        a[5].set(title="主峰放大与功率半高宽", xlabel=distance_label, ylabel="归一化幅度")
        peak = current.peak_distance_m if np.isfinite(current.peak_distance_m) else self.model.cfg.distance_m
        lower = min(self.model.cfg.distance_m, peak) - 3.5
        upper = max(self.model.cfg.distance_m, peak) + 3.5
        a[5].set_xlim(lower, upper)
        a[5].set_ylim(0, 1.09)
        a[5].axhline(1 / np.sqrt(2), color="#cbd5e1", lw=0.9, label="_nolegend_")
        for analysis, color in ((ref, self.IDEAL_COLOR), (current, self.CURRENT_COLOR)):
            if np.isfinite(analysis.width_3db_m):
                a[5].plot([analysis.left_crossing_m, analysis.right_crossing_m],
                          [1 / np.sqrt(2)] * 2, "o-", color=color, lw=2, ms=4)
        for path_d in np.asarray(self.model.true_path_distances_m).ravel():
            if np.isfinite(path_d):
                a[5].axvline(path_d, color="#059669", ls=":", lw=0.8)
        for ax in a[:4]:
            ax.set_xlim(float(f[0]), float(f[-1]))
        title = PRESETS.get(self.model.preset_name, "手动 / CSV 配置")
        self.subtitle.set_text(
            f"{self.model.cfg.n_tones} 个频点 × {self.model.cfg.df_hz/1e6:g} MHz  |  "
            f"参考距离 {self.model.cfg.distance_m:g} m  |  {title}  |  频点编号从 0 开始")
        self._update_status()
        self.fig.canvas.draw_idle()

    def save_snapshot(self, output_dir: Path | str | None = None) -> Path:
        folder = _new_output_folder(Path(output_dir or self.output_dir))
        self.model.save_csv(folder / "iq_edits.csv")
        current = self.current
        np.savez_compressed(folder / "iq_cir_data.npz",
                            tone_index=self.model.indices,
                            frequency_hz=self.model.frequencies_hz,
                            amplitude_scale=self.model.amplitude_scale,
                            phase_delta_deg=self.model.phase_delta_deg,
                            ideal_iq=self.model.ideal_iq, edited_iq=self.model.iq,
                            tau_s=current.tau_s, distance_m=current.distance_m,
                            ideal_cir=self.reference.cir, edited_cir=current.cir)
        summary = {"config": asdict(self.model.cfg),
                   "preset": self.model.preset_name,
                   "ideal": _metrics(self.reference), "edited": _metrics(current),
                   "width_definition": "power half maximum; amplitude threshold 1/sqrt(2)",
                   "note": "Synthetic equivalent-delay IQ model; zero padding does not add resolution."}
        (folder / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                         allow_nan=False), encoding="utf-8")
        self.fig.savefig(folder / "iq_cir_interactive.png", dpi=130)
        return folder

    def _save_snapshot_clicked(self) -> None:
        folder = self.save_snapshot()
        self._message = "图像及数据已保存：" + folder.name
        self._update_status()
        self.fig.canvas.draw_idle()


def save_preset_overview(output_dir: Path | str, cfg: Config | None = None) -> Path:
    """独立导出三类预设的静态 3×3 总览，同时保存每个案例的数据。"""
    configure_fonts()
    folder = _new_output_folder(Path(output_dir), "presets")
    model = IQModel(cfg or CONFIG)
    ideal = model.analyze(model.ideal_iq)
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 9.2))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.10, hspace=0.55, wspace=0.29)
    fig.suptitle("理想 IQ 与三类异常 IQ / CIR 对比", x=0.06, ha="left", fontsize=21, weight="bold")
    fig.text(0.06, 0.919, "Python 数值输出 · 合成数据 · 频域幅度统一尺度 · CIR 各自按峰值归一化",
             color="#64748b", fontsize=11)
    titles = ["频域 IQ 幅度", "去除各自最佳线性相位后的残差", "IFFT CIR 幅度"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, pad=16, weight="bold", fontsize=12)
    summary = {"config": asdict(model.cfg), "ideal": _metrics(ideal), "cases": {}}
    for row, (key, color) in enumerate(zip(("edge_gaussian", "quadratic_phase", "close_multipath"),
                                          ("#2563eb", "#c2410c", "#078070"))):
        model.apply_preset(key)
        data = model.analyze()
        f = model.f_offset_hz / 1e6
        amp, phase, cir = axes[row]
        amp.plot(f, abs(model.ideal_iq), "--", color="#94a3b8", lw=1.6)
        amp.plot(f, abs(model.iq), color=color, lw=2)
        amp.set_ylabel(PRESETS[key], fontsize=11, weight="bold")
        amp.set_ylim(0, 1.15)
        phase.plot(f, ideal.phase_residual_rad, "--", color="#94a3b8", lw=1.6)
        phase.plot(f, data.phase_residual_rad, color=color, lw=2)
        phase.set_ylabel("残差 / rad")
        if np.nanmax(abs(data.phase_residual_rad)) < 1e-9:
            phase.set_ylim(-0.2, 0.2)
        cir.plot(ideal.distance_m, ideal.normalized_amplitude, "--", color="#94a3b8", lw=1.6,
                 label=f"理想：{ideal.width_3db_m:.3f} m")
        cir.plot(data.distance_m, data.normalized_amplitude, color=color, lw=2,
                 label=f"本例：{data.width_3db_m:.3f} m")
        cir.axhline(1 / np.sqrt(2), color="#cbd5e1", lw=0.8)
        for distance in np.asarray(model.true_path_distances_m).ravel():
            cir.axvline(distance, color="#059669", ls=":", lw=0.8)
        cir.set_xlim(model.cfg.distance_m - 4, model.cfg.distance_m + 4)
        cir.set_ylim(0, 1.1)
        cir.set_ylabel("各自峰值归一化幅度")
        cir.legend(loc="upper right", fontsize=9, frameon=False)
        cir.set_xlabel("双程等效距离 / m" if model.cfg.round_trip else "单程距离 / m")
        for ax in (amp, phase):
            ax.set_xlabel("相对中心频率 / MHz")
            ax.set_xlim(f[0], f[-1])
        for ax in (amp, phase, cir):
            ax.grid(axis="y", color="#e2e8f0")
        model.save_csv(folder / f"{key}_edits.csv")
        np.savez_compressed(folder / f"{key}_iq_cir.npz",
                            frequency_hz=model.frequencies_hz, ideal_iq=model.ideal_iq,
                            edited_iq=model.iq, distance_m=data.distance_m,
                            ideal_cir=ideal.cir, edited_cir=data.cir)
        summary["cases"][key] = _metrics(data)
    fig.text(0.06, 0.030, "功率半高宽对应幅度 0.707；补零只增加采样密度，不增加物理分辨率。",
             fontsize=10, color="#64748b")
    fig.savefig(folder / "three_cases_overview.png", dpi=150)
    plt.close(fig)
    (folder / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                     allow_nan=False), encoding="utf-8")
    return folder


def main(argv: list[str] | None = None) -> int:
    # Windows 某些终端默认使用 cp1252，显式使用 UTF-8 保留中文输出。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-gui", action="store_true", help="只导出 PNG、CSV、NPZ 和 JSON，不弹出窗口")
    parser.add_argument("--preset", choices=list(PRESETS), default=INITIAL_PRESET, help="启动时的预设")
    parser.add_argument("--load-csv", type=Path, help="启动时加载逐频点倍率/相位配置")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="导出目录")
    args = parser.parse_args(argv)
    if args.no_gui and matplotlib.get_backend().lower() != "agg":
        plt.switch_backend("Agg")
    model = IQModel(CONFIG)
    model.apply_preset(args.preset)
    for index, edits in POINT_EDITS.items():
        model.set_point(index, **edits)
    if args.load_csv is not None:
        model.load_csv(args.load_csv)
    explorer = IQExplorer(model, args.output_dir)
    if args.no_gui:
        snapshot = explorer.save_snapshot()
        overview = save_preset_overview(args.output_dir, model.cfg)
        plt.close(explorer.fig)
        print(f"当前配置图像与数据：{snapshot}")
        print(f"三类预设对比图：{overview}")
    else:
        print("操作：选频点 → 拖动幅度/相位滑块；输入精确值后按回车。")
        print("可保存/读取 CSV，逐点配置可在表格编辑器中修改。")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
