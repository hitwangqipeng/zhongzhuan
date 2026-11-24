"""Batch inference on already-prepared, canonical combined IQ."""
import numpy as np
from .frontend import covariance_features, sdk_time_window
from .model import FrozenTiModel


def infer_iq(iq, modes=("none", "sdk"), distance_correction_m=0., batch_size=512,
             model=None, *, stream_ids=None, velocity_mps=None, history_alpha=0.5,
             min_snr_db=9.0):
    """For sdk_hist, supply chronologically ordered rows and a key per stream.

    History lives for one call, across its internal batch boundaries. Missing
    velocity assumes stationary observations. The SDK-style gate masks only
    the reported distance; the ungated NN output is retained for diagnosis.
    """
    z = np.asarray(iq, np.complex128)
    if z.ndim != 2 or z.shape[1] != 80 or len(z) == 0 or not np.isfinite(z).all():
        raise ValueError("Expected finite nonempty N x80 combined IQ")
    modes = list(modes)
    if not modes or len(set(modes)) != len(modes) or any(v not in ("none", "sdk", "sdk_hist") for v in modes):
        raise ValueError("modes must contain none, sdk and/or sdk_hist, without duplicates")
    if not np.isfinite(distance_correction_m) or batch_size < 1:
        raise ValueError("Invalid distance correction or batch size")
    model = FrozenTiModel() if model is None else model
    if "sdk_hist" in modes:
        if not 0 <= history_alpha <= 1 or not np.isfinite(min_snr_db):
            raise ValueError("Invalid history_alpha or min_snr_db")
        if stream_ids is None:
            if history_alpha < 1:
                raise ValueError("sdk_hist requires one stream ID per row when history is enabled")
            stream_ids = (None,) * len(z)
        else:
            stream_ids = tuple(stream_ids)
            if len(stream_ids) != len(z):
                raise ValueError("stream_ids must contain one hashable key per row")
            try:
                set(stream_ids)
            except TypeError as exc:
                raise ValueError("stream_ids must contain one hashable key per row") from exc
        velocity_mps = np.broadcast_to(np.asarray(0. if velocity_mps is None else velocity_mps, float), (len(z),))
        if not np.isfinite(velocity_mps).all():
            raise ValueError("velocity_mps must be finite")
    result = {}
    for mode in modes:
        raw = np.empty(len(z), np.float32)
        states = np.zeros(len(z), np.int32)
        if mode == "sdk_hist":
            history = {}
            snr = np.zeros(len(z), np.float32)
            valid = np.zeros(len(z), bool)
            used_history = np.zeros(len(z), bool)
        for start in range(0, len(z), batch_size):
            chunk = z[start:start+batch_size]
            rows = slice(start, start+len(chunk))
            if mode == "sdk":
                chunk, state = sdk_time_window(chunk)
                states[rows] = state
            elif mode == "sdk_hist":
                # IFFT is linear, so mixing unnormalized correlogram power
                # here is equivalent to mixing the frequency autocorrelations
                # before the IFFT for this single-path adapter.
                power = (abs(np.fft.ifft(chunk, 256, axis=1))**2).astype(np.float32)
                if history_alpha < 1:
                    for i, key in enumerate(stream_ids[rows]):
                        previous = history.get(key)
                        if previous is not None and velocity_mps[start+i] == 0:
                            power[i] = history_alpha * power[i] + (1 - history_alpha) * previous
                            used_history[start+i] = True
                        history[key] = power[i].copy()
                chunk, states[rows], snr[rows], valid[rows] = sdk_time_window(
                    chunk, power=power, return_diagnostics=True
                )
            raw[rows] = model.predict(covariance_features(chunk))
        result[f"raw_model_{mode}_m"] = raw
        result[f"distance_{mode}_m"] = raw.astype(float)-float(distance_correction_m)
        if mode == "sdk":
            result["peak_bin_sap"] = states
        elif mode == "sdk_hist":
            valid &= snr > min_snr_db
            result["distance_sdk_hist_m"][~valid] = np.nan
            result["peak_bin_sap_sdk_hist"] = states
            result["snr_sdk_hist_db"] = snr
            result["valid_sdk_hist"] = valid
            result["used_history_sdk_hist"] = used_history
    return result


def error_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    keep = np.isfinite(prediction) & np.isfinite(truth)
    n = int(keep.sum())
    base = dict(n_input=len(truth), n_evaluated=n, missing_reference_n=int((~np.isfinite(truth)).sum()))
    if not n:
        return dict(base, mae_m="", error_p5_m="", abs_error_p5_m="", cep95_m="", bias_m="", under3_pct="")
    e = prediction[keep]-truth[keep]
    return dict(base, mae_m=float(abs(e).mean()), error_p5_m=float(np.quantile(e, .05)),
        abs_error_p5_m=float(np.quantile(abs(e), .05)), cep95_m=float(np.quantile(abs(e), .95)),
        bias_m=float(e.mean()), under3_pct=float(100*np.mean(e < -3)))
