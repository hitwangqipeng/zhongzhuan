"""Physics and interaction checks for the IQ explorer; no GUI window is opened."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ble_iq_model import Config, IQModel

CHECK_OUTPUT = Path(__file__).resolve().parent / 'outputs' / 'ble_iq_python' / 'verification'
CHECK_OUTPUT.mkdir(parents=True, exist_ok=True)


class IQPhysicsChecks(unittest.TestCase):
    def test_single_tone_edit_matches_analytic_impulse_perturbation(self):
        model = IQModel()
        base = model.analyze()
        tone, gain, phase_deg = 13, 0.35, 57.0
        model.set_point(tone, amplitude_scale=gain, phase_delta_deg=phase_deg)
        changed = model.analyze()
        delta_iq = model.ideal_iq[tone] * (
            gain * np.exp(1j * np.deg2rad(phase_deg)) - 1
        )
        expected = delta_iq * np.exp(
            2j * np.pi * model.f_offset_hz[tone] * base.tau_s
        ) / model.cfg.n_tones
        np.testing.assert_allclose(
            changed.cir - base.cir, expected, atol=1e-14, rtol=1e-12
        )
        untouched = np.arange(model.cfg.n_tones) != tone
        np.testing.assert_allclose(model.iq[untouched], model.ideal_iq[untouched])

    def test_reference_width_distance_and_global_scaling_invariances(self):
        model = IQModel()
        base = model.analyze()
        self.assertAlmostEqual(base.peak_distance_m, 8.0, places=4)
        self.assertAlmostEqual(base.width_3db_ns, 11.0744085, places=3)
        model.set_range(0, model.cfg.n_tones - 1,
                        amplitude_scale=2.5, phase_delta_deg=123)
        changed = model.analyze()
        np.testing.assert_allclose(changed.normalized_amplitude,
                                   base.normalized_amplitude, atol=1e-14)
        self.assertAlmostEqual(changed.peak_amplitude / base.peak_amplitude,
                               2.5, places=12)
        self.assertAlmostEqual(changed.width_3db_m, base.width_3db_m, places=10)

    def test_presets_separate_shift_broadening_and_missing_points(self):
        model = IQModel()
        base = model.analyze()
        model.apply_preset('linear_phase')
        shifted = model.analyze()
        self.assertAlmostEqual(shifted.peak_distance_m - base.peak_distance_m,
                               1.2, places=4)
        self.assertAlmostEqual(shifted.width_3db_m, base.width_3db_m, places=4)
        for preset in ('edge_gaussian', 'quadratic_phase', 'close_multipath'):
            model.apply_preset(preset)
            self.assertGreater(model.analyze().width_3db_m, base.width_3db_m)
        model.apply_preset('center_missing')
        self.assertLess(model.analyze().width_3db_m, base.width_3db_m)
        self.assertEqual(np.count_nonzero(model.amplitude_scale == 0), 10)

    def test_zero_and_single_tone_do_not_report_a_fake_width(self):
        model = IQModel()
        model.set_range(0, model.cfg.n_tones - 1, amplitude_scale=0)
        zero = model.analyze()
        self.assertTrue(np.isnan(zero.width_3db_m))
        self.assertTrue(np.isnan(zero.peak_distance_m))
        self.assertTrue(np.all(zero.normalized_amplitude == 0))
        model.set_point(12, amplitude_scale=1)
        single = model.analyze()
        self.assertTrue(np.isnan(single.width_3db_m))
        np.testing.assert_allclose(single.amplitude,
                                   np.full(model.cfg.nfft, 1/model.cfg.n_tones),
                                   atol=1e-14)

    def test_invalid_edits_are_atomic(self):
        model = IQModel()
        before = model.iq.copy()
        invalid = [
            lambda: model.set_point(-1, amplitude_scale=1),
            lambda: model.set_point(80, phase_delta_deg=0),
            lambda: model.set_point(3.5, phase_delta_deg=0),
            lambda: model.set_point(True, phase_delta_deg=0),
            lambda: model.set_point(3, amplitude_scale=-1),
            lambda: model.set_point(3, amplitude_scale=0.5, phase_delta_deg=np.nan),
            lambda: model.set_range(5, 2, amplitude_scale=0),
            lambda: model.set_range(0, 79, phase_delta_deg=np.inf),
        ]
        for edit in invalid:
            with self.assertRaises((ValueError, TypeError, IndexError)):
                edit()
            np.testing.assert_array_equal(model.iq, before)

    def test_csv_round_trip_shuffled_indices_and_bad_input(self):
        model = IQModel()
        model.apply_preset('quadratic_phase')
        model.set_point(10, amplitude_scale=0.3, phase_delta_deg=72)
        with tempfile.TemporaryDirectory(prefix='ble_iq_check_', dir=CHECK_OUTPUT) as temp:
            self.assertTrue(Path(temp).resolve().is_relative_to(CHECK_OUTPUT.resolve()))
            path = Path(temp) / 'edits.csv'
            model.save_csv(path)
            with path.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                rows = list(reader)
            with path.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows[::-1])
            restored = IQModel()
            restored.load_csv(path)
            np.testing.assert_allclose(restored.iq, model.iq, atol=1e-12)
            before = restored.iq.copy()
            rows[0]['tone_index'] = rows[1]['tone_index']
            with path.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises((ValueError, TypeError, IndexError)):
                restored.load_csv(path)
            np.testing.assert_array_equal(restored.iq, before)

    def test_one_way_and_round_trip_coordinate_conventions(self):
        for round_trip in (False, True):
            model = IQModel(Config(round_trip=round_trip))
            result = model.analyze()
            self.assertAlmostEqual(result.peak_distance_m, 8.0, places=4)
            expected_factor = 2 if round_trip else 1
            self.assertAlmostEqual(result.width_3db_m,
                                   result.width_3db_ns * 1e-9 * model.cfg.c /
                                   expected_factor, places=10)


class IQInteractionChecks(unittest.TestCase):
    def test_gui_callbacks_and_saved_snapshot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from ble_iq_explorer import IQExplorer

        app = IQExplorer(IQModel())
        try:
            app.select_point(10)
            app.set_selected(amplitude_scale=0.5, phase_delta_deg=45)
            self.assertAlmostEqual(app.model.amplitude_scale[10], 0.5)
            self.assertAlmostEqual(app.model.phase_delta_deg[10], 45)
            app.apply_range(20, 23, amplitude_scale=0.25, phase_delta_deg=-30)
            np.testing.assert_allclose(app.model.amplitude_scale[20:24], 0.25)
            np.testing.assert_allclose(app.model.phase_delta_deg[20:24], -30)
            app.apply_preset('quadratic_phase')
            self.assertTrue(np.allclose(app.model.amplitude_scale, 1))
            app.set_selected(amplitude_scale=5, phase_delta_deg=540)
            self.assertAlmostEqual(app.model.amplitude_scale[10], 5)
            self.assertAlmostEqual(app.model.phase_delta_deg[10], 540)
            with tempfile.TemporaryDirectory(prefix='ble_iq_gui_check_', dir=CHECK_OUTPUT) as temp:
                self.assertTrue(Path(temp).resolve().is_relative_to(CHECK_OUTPUT.resolve()))
                app.save_snapshot(Path(temp))
                self.assertTrue(list(Path(temp).rglob('*.png')))
                self.assertTrue(list(Path(temp).rglob('*.csv')))
                self.assertTrue(list(Path(temp).rglob('*.npz')))
            app.apply_range(0, 79, amplitude_scale=0)
            app.refresh()
            self.assertTrue(np.isnan(app.model.analyze().width_3db_m))
        finally:
            plt.close(app.fig)


if __name__ == '__main__':
    unittest.main(verbosity=2)
