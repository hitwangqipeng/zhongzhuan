"""Causal per-path IQ repair with reference trust and proposal acceptance gates.

Only NumPy/SciPy are required. One instance per anchor/path stream; the same
configuration is shared across streams. No range truth, folder label, future
frame, path-ID feature or calibrated reliability probability is used.

The default ranging callback reproduces the existing distance-sorted 0.42 /
rising-peak rule. Supply distance_estimator=your_callable to use another
estimator; recalibrate acceptance settings when changing that callback.
History contains ORIGINAL observations, including rejected frames. Repaired
IQ is never fed back. This method addresses transient localized distortions,
not persistent wrong channels or an unobserved physical first path.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class ReferenceRepairConfig:
    history_size: int = 5
    min_history: int = 3
    group_size: int = 4
    history_max_loo_nmse: float = .15
    target_max_history_disagreement: float = .35
    current_max_outside_nmse: float = .15
    min_group_excess_ratio: float = 4.
    min_prediction_agreement: float = .6
    min_history_distance_gain_m: float = .5
    prediction_tolerance_m: float = 299792458/(2*80e6)
    max_history_age_s: float = 2.
    max_gap_s: float = 1.
    enabled: bool = True


@dataclass
class IQRepairProposal:
    original_iq: np.ndarray
    proposed_iq: np.ndarray
    group: int | None
    diagnostics: dict


@dataclass
class IQRepairResult:
    iq: np.ndarray
    accepted: bool
    reference_trusted: bool
    reason: str
    group: int | None
    diagnostics: dict
    proposed_iq: np.ndarray | None = None


def _validate(config):
    for name in ['history_size','min_history','group_size']:
        value=getattr(config,name)
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
            raise ValueError(f'{name} must be an integer')
    if not (3 <= config.min_history <= config.history_size):
        raise ValueError('Require 3 <= min_history <= history_size')
    if config.group_size<1 or 80%config.group_size:
        raise ValueError('group_size must divide 80')
    for name in ['history_max_loo_nmse','target_max_history_disagreement','current_max_outside_nmse',
                 'min_group_excess_ratio','prediction_tolerance_m','max_history_age_s','max_gap_s']:
        value=getattr(config,name)
        if not np.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive and finite')
    if not 0<config.min_prediction_agreement<=1:raise ValueError('min_prediction_agreement must be in (0,1]')
    if not np.isfinite(config.min_history_distance_gain_m) or config.min_history_distance_gain_m<0:
        raise ValueError('min_history_distance_gain_m must be nonnegative and finite')
    if not isinstance(config.enabled,(bool,np.bool_)):raise ValueError('enabled must be boolean')


def _iq(signal):
    z=np.asarray(signal,dtype=np.complex128)
    if z.shape!=(80,) or not np.isfinite(z).all():raise ValueError('Expected 80 finite complex IQ samples')
    return z.copy()


def traditional_distance(iq80):
    """Existing 4096-IFFT rule, 1 MHz tones, two-way mapping, -0.94 m offset.

    Local maxima are measured on the complete circular CIR. Retain raw 0..60m,
    parabolically refine positions and then select first rising peak >0.42.
    A missing valid selection is NaN; no synthetic candidate is fabricated.
    """
    period=299792458/2e6;n=4096;step=period/n
    a=abs(np.fft.ifft(iq80,n));stop=int(60/step)+1
    maximum=float(a[:stop].max())
    if maximum<=np.finfo(float).tiny:return np.nan
    pp=find_peaks(np.r_[a[-1],a,a[0]])[0]-1
    left,center,right=a[(pp-1)%n],a[pp],a[(pp+1)%n]
    den=left-2*center+right
    off=np.divide(.5*(left-right),den,out=np.zeros_like(center),where=abs(den)>1e-20)
    raw=(pp+np.clip(off,-.5,.5))*step
    keep=(raw>=0)&(raw<=60);raw=raw[keep];height=center[keep]/maximum
    if not len(raw):return np.nan
    cond=(height>.42)&np.r_[False,height[1:]>height[:-1]]
    ids=np.flatnonzero(cond)
    return float(max(raw[ids[0]]-.94,0)) if len(ids) else np.nan


def _shape(history):
    unit=history/np.maximum(np.linalg.norm(history,axis=1)[:,None],1e-12)
    medoid=unit[np.argmax(abs(unit@unit.conj().T).sum(1))]
    aligned=unit*np.exp(-1j*np.angle(unit@medoid.conj()))[:,None]
    return np.median(aligned.real,axis=0)+1j*np.median(aligned.imag,axis=0)


def _fit(signal,shape,mask=None):
    """Robust common complex gain, identical to the earlier past-only baseline."""
    good=np.ones(80,bool) if mask is None else np.asarray(mask,bool)
    weights=good.astype(float)
    alpha=0j
    for _ in range(6):
        alpha=np.sum(weights*shape.conj()*signal)/max(float(np.sum(weights*abs(shape)**2)),1e-20)
        residual=abs(signal-alpha*shape)
        scale=max(float(np.median(residual[good])),float(np.median(abs(signal[good])))*.01,1e-9)
        weights=good*np.minimum(1,2*scale/np.maximum(residual,1e-9))
    return alpha*shape


def propose_iq_repair(iq80, history: Sequence[np.ndarray], *,
                      config: ReferenceRepairConfig | None=None,
                      distance_estimator: Callable | None=None) -> IQRepairProposal:
    """Compute ONE causal candidate and IQ-only diagnostics before gate decisions.

    history must be chronologically preceding samples of this same stream.
    This function cannot verify caller ordering; the streaming wrapper does.
    The proposed four values reproduce the earlier ungated past-only method.
    """
    cfg=config or ReferenceRepairConfig();_validate(cfg)
    estimate=distance_estimator or traditional_distance
    signal=_iq(iq80)
    previous=np.asarray([_iq(h) for h in list(history)[-cfg.history_size:]])
    d=dict(history_count=len(previous),raw_distance_m=float(estimate(signal)),proposal_available=False)
    if len(previous)<cfg.min_history:
        return IQRepairProposal(signal,signal.copy(),None,d)
    if np.linalg.norm(signal)<=1e-12 or np.any(np.linalg.norm(previous,axis=1)<=1e-12):
        return IQRepairProposal(signal,signal.copy(),None,d)
    reference=_fit(signal,_shape(previous))
    residual=signal-reference
    errors=(abs(residual)**2).reshape(-1,cfg.group_size).sum(1)
    group=int(errors.argmax());indices=np.arange(group*cfg.group_size,(group+1)*cfg.group_size)
    outside=np.ones(80,bool);outside[indices]=False
    total=max(float(np.sum(abs(signal)**2)),1e-20)
    outside_energy=max(float(np.sum(abs(signal[outside])**2)),total*1e-8)
    loo=[]
    for j in range(len(previous)):
        template=_shape(np.delete(previous,j,axis=0))
        prediction=_fit(previous[j],template)
        loo.append(float(np.sum(abs(previous[j]-prediction)**2)/max(float(np.sum(abs(previous[j])**2)),1e-20)))
    proposed=signal.copy();proposed[indices]=reference[indices]
    reference_energy=max(float(np.sum(abs(reference[indices])**2)),total*cfg.group_size/80*.01,1e-20)
    group_disagreement=[];candidate_distances=[]
    for h in previous:
        individual=_fit(signal,h,outside)
        group_disagreement.append(float(np.sum(abs(individual[indices]-reference[indices])**2)/reference_energy))
        alternate=signal.copy();alternate[indices]=individual[indices]
        candidate_distances.append(float(estimate(alternate)))
    candidate_distances=np.asarray(candidate_distances)
    proposed_distance=float(estimate(proposed))
    history_distances=np.array([estimate(h) for h in previous],float)
    finite_history=history_distances[np.isfinite(history_distances)]
    if np.isfinite(proposed_distance) and np.isfinite(d['raw_distance_m']) and len(finite_history)>=cfg.min_history:
        gain=float(np.median(abs(d['raw_distance_m']-finite_history)-abs(proposed_distance-finite_history)))
    else:gain=np.nan
    agreement=float(np.mean(np.isfinite(candidate_distances)&(abs(candidate_distances-proposed_distance)<=cfg.prediction_tolerance_m)))
    typical=float(np.median(np.delete(errors,group)))
    d.update(proposal_available=True,group=group,first_tone=int(indices[0]),last_tone=int(indices[-1]),
        proposed_distance_m=proposed_distance,history_loo_nmse_q75=float(np.quantile(loo,.75)),
        target_history_disagreement_q75=float(np.quantile(group_disagreement,.75)),
        current_outside_nmse=float(np.sum(abs(residual[outside])**2)/outside_energy),
        group_excess_ratio=float(errors[group]/max(typical,total*1e-12)),
        group_residual_fraction=float(errors[group]/max(float(errors.sum()),total*1e-12)),
        prediction_agreement=agreement,history_distance_gain_m=gain,
        history_distance_median_m=float(np.median(finite_history)) if len(finite_history) else np.nan,
        ensemble_distances_m=candidate_distances.tolist())
    return IQRepairProposal(signal,proposed,group,d)


def decide_repair(diagnostics,config: ReferenceRepairConfig | None=None):
    """Return (accepted, reference_trusted, reason), with explicit abstention.

    Scores are diagnostics, not probabilities. Distance checks compare only
    observed/predicted historical distances; true range is never an input.
    """
    cfg=config or ReferenceRepairConfig();_validate(cfg);d=diagnostics
    if not cfg.enabled:return False,False,'disabled'
    if not d.get('proposal_available',False):return False,False,'insufficient_valid_history'
    for key,maximum,reason in [
        ('history_loo_nmse_q75',cfg.history_max_loo_nmse,'reference_history_unstable'),
        ('target_history_disagreement_q75',cfg.target_max_history_disagreement,'reference_target_uncertain'),
        ('current_outside_nmse',cfg.current_max_outside_nmse,'current_broadband_change')]:
        if not np.isfinite(d.get(key,np.nan)) or d[key]>maximum:return False,False,reason
    if d['group_excess_ratio']<cfg.min_group_excess_ratio:return False,True,'no_isolated_group_anomaly'
    if d['prediction_agreement']<cfg.min_prediction_agreement:return False,True,'repair_predictions_disagree'
    if not np.isfinite(d['history_distance_gain_m']):return False,True,'distance_unavailable'
    if d['history_distance_gain_m']<cfg.min_history_distance_gain_m:return False,True,'no_history_consistency_gain'
    return True,True,'accepted'


class CausalIQReferenceRepair:
    """One independent raw-IQ history per instance, all gates shared by config."""
    def __init__(self,config=None,*,distance_estimator=None):
        self.config=config or ReferenceRepairConfig();_validate(self.config)
        self.distance_estimator=distance_estimator or traditional_distance
        self._history=deque(maxlen=self.config.history_size)
        self._last_timestamp=None

    def reset(self):
        self._history.clear();self._last_timestamp=None

    def history_snapshot(self):
        return [z.copy() for _,z in self._history]

    def process(self,iq80,*,timestamp_s:float | None=None) -> IQRepairResult:
        signal=_iq(iq80);cfg=self.config;reset=False
        if self._history and ((timestamp_s is None)!=(self._last_timestamp is None)):
            self.reset();reset=True
        if timestamp_s is not None:
            if not np.isfinite(timestamp_s):raise ValueError('timestamp_s must be finite')
            if self._last_timestamp is not None and (timestamp_s<=self._last_timestamp or timestamp_s-self._last_timestamp>cfg.max_gap_s):
                self.reset();reset=True
            while self._history and self._history[0][0] is not None and timestamp_s-self._history[0][0]>cfg.max_history_age_s:
                self._history.popleft()
        if cfg.enabled:
            proposal=propose_iq_repair(signal,self.history_snapshot(),config=cfg,distance_estimator=self.distance_estimator)
            accepted,trusted,reason=decide_repair(proposal.diagnostics,cfg)
            diagnostics=proposal.diagnostics.copy();diagnostics.update(reference_trusted=trusted,accepted=accepted,reason=reason,history_reset=reset)
            result=IQRepairResult(proposal.proposed_iq.copy() if accepted else signal.copy(),accepted,trusted,reason,proposal.group,diagnostics,proposal.proposed_iq.copy())
        else:
            diagnostics=dict(history_count=len(self._history),proposal_available=False,reference_trusted=False,
                accepted=False,reason='disabled',history_reset=reset)
            result=IQRepairResult(signal.copy(),False,False,'disabled',None,diagnostics,signal.copy())
        self._history.append((timestamp_s,signal.copy()))
        self._last_timestamp=timestamp_s
        return result


def config_dict(config=None):
    return asdict(config or ReferenceRepairConfig())
