"""Distance-disjoint, independent-path Qualcomm-inspired representation study.

Four controlled cells: power / complex CIR x end-to-end / weighted RnC.
Patent topology and two-stage idea, with explicit implementation assumptions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from experiment_real_car_remnet import OUT as PREVIOUS, DISTANCES, parse_source, sample_indices

OUT = PREVIOUS.parent / 'qualcomm_contrastive_20261111'
CELLS = {'A': ('power', 'e2e'), 'B': ('power', 'rnc'),
         'C': ('complex', 'e2e'), 'D': ('complex', 'rnc')}
NFFT = 512
SEED = 20263111
ENC_EPOCHS = 48
BATCH = 128
TEMPERATURE = .15
COSTS = (1, 9, 19)


def protocol():
    OUT.mkdir(exist_ok=True)
    p = dict(title='Qualcomm-inspired weighted RnC / direct distance regression',
        cells=CELLS, source='https://patents.google.com/patent/US20260222246A1/en',
        rnc_reference='https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py',
        reproduction='method reimplementation, not author code, weights, or reported performance',
        data=dict(files=64, samples=113928, frames=28482, source=str(PREVIOUS),
                  truths=32, training_frames_per_log=80, raw_source_read_only=True),
        input=dict(nfft=NFFT, native_tones=80, order='same fftshifted grid for all four cells',
            power='log10(max(|CIR|^2/max|CIR|^2,1e-6))/6, one channel',
            complex='Re(CIR)/max|CIR|, Im(CIR)/max|CIR|, two channels',
            prohibited='no IDs, metadata, raw distance, true distance, or folder state as model inputs',
            physical='already two-way IQ; no repeated multiplication, square root, or phase calibration',
            grid_choice='512 for both representations to isolate phase; patent 256 is an example'),
        split='reuse original fixed eight whole-distance outer folds and validation distances; no log/frame overlap',
        samples='one antenna path is one sample; no antenna fusion anywhere',
        target='direct 3D distance, standardized using training-only mean/std',
        network=dict(stem='Conv1d(in,16,7,stride=4,padding=3), ReLU',
            residual_blocks=2, block='Conv3 stride2, ReLU, Conv3 stride1 + Conv1 stride2 skip, ReLU',
            feature='AdaptiveAvgPool1d(16), flatten, Linear(256,64), L2 normalize',
            head='Linear(64,32), ReLU, Linear(32,1)',
            initialization='same per-fold seed, no previous trained encoder'),
        augmentation=dict(views=2, noise_relative_rms=.01, noise='independent complex Gaussian per tone',
            phase='one uniform [-pi,pi] rotation per path across all tones',
            same_for_e2e_and_rnc=True, no_frequency_ramp=True, no_crop=True),
        encoder_training=dict(max_epochs=ENC_EPOCHS, min_epochs=16, patience=12, batch=BATCH,
            lr=.001, weight_decay=.0001, temperature=TEMPERATURE, torch_threads=1,
            sampling='same shuffled selected paths for all cells, both views per path',
            validation='fixed balanced 128-path subsets across four validation truths; RnC loss for encoder; MSE for E2E'),
        contrastive=dict(similarity='negative Euclidean distance of normalized features',
            label_distance='absolute 3D distance difference',
            candidates='k!=i and |yi-yk| >= |yi-yj|; includes equal-distance ties',
            weights='1 + |yi-yk|/s; s = median nonzero pair difference of unique training truths',
            assumption='positive unit floor and training-only scale close the underspecified patent weights',
            implementation='sorted suffix logsumexp, identical-distance ties share denominator',
            validation_has_no_gradient=True),
        heads=dict(mse='E2E primary output or frozen RnC encoder + independently fitted MSE head',
            under='freeze each selected encoder, train separate cost 1/9/19 heads',
            loss='[lambda max(truth-pred,0) + max(pred-truth,0)]/(lambda+1)',
            optimizer='full-batch AdamW lr=.003 weight_decay=.0001',
            max_epochs=600, min_epochs=60, patience=80,
            selection='each head by its own validation loss; no test selection'),
        output='unbounded raw distance primary; zero-floor secondary explicitly marked',
        no_temporal_smoothing=True, no_gate=True, no_margin=True, seed=SEED)
    p = json.loads(json.dumps(p))
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8')) == p, 'Protocol changed after initialization'
    else:
        path.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding='utf8')
    return p


def prepare():
    protocol()
    meta = pd.read_csv(PREVIOUS / 'paths.csv.gz')
    audit = pd.read_csv(PREVIOUS / 'source_audit.csv')
    assert len(meta) == 113928 and len(audit) == 64
    dest = OUT / 'raw_iq.npy'
    if dest.exists():
        z = np.load(dest, mmap_mode='r')
        assert z.shape == (len(meta), 80) and z.dtype == np.complex64
        print('PREPARED CACHE EXISTS', z.shape, flush=True)
        return
    z = np.lib.format.open_memmap(dest.with_suffix('.partial.npy'), mode='w+',
                                 dtype=np.complex64, shape=(len(meta), 80))
    old_power = np.load(PREVIOUS / 'spectrum.npy', mmap_mode='r')
    max_delta = 0.
    for file_id, entry in audit.iterrows():
        raw = Path(entry.file)
        md, iq, bad, sha = parse_source(raw)
        ids = np.flatnonzero(meta.file_id.eq(file_id).to_numpy())
        assert sha == entry.sha256 and len(ids) == len(md)
        assert np.array_equal(md[['epoch', 'key', 'ts', 'path']].to_numpy(),
                              meta.iloc[ids][['epoch', 'key', 'ts', 'path']].to_numpy())
        values = iq[:, :, 0].astype(np.float32) + 1j * iq[:, :, 1].astype(np.float32)
        z[ids] = values
        for start in range(0, len(ids), 128):
            pick = ids[start:start+128]
            c = np.fft.fftshift(np.fft.ifft(values[start:start+128].astype(np.complex128), n=NFFT), axes=1)
            power = np.abs(c)**2
            x = (np.log10(np.maximum(power/power.max(1, keepdims=True), 1e-6))/6).astype(np.float32)
            delta = float(np.max(np.abs(x-old_power[pick])))
            assert delta < 1e-6, delta
            max_delta = max(max_delta, delta)
        print('PREP', file_id+1, len(audit), len(ids), flush=True)
    z.flush()
    del z
    dest.with_suffix('.partial.npy').replace(dest)
    (OUT/'input_audit.json').write_text(json.dumps(dict(samples=len(meta), native_tones=80,
        power_cache_max_abs_difference=max_delta, source_hashes_verified=64,
        metadata_order_verified=True), indent=2), encoding='utf8')


def represent(z, kind):
    c = torch.fft.fftshift(torch.fft.ifft(z, n=NFFT, dim=-1), dim=-1)
    mag = c.abs()
    peak = mag.amax(-1, keepdim=True).clamp_min(1e-12)
    if kind == 'power':
        return (torch.log10((mag/peak).square().clamp_min(1e-6))/6).unsqueeze(1)
    c = c/peak
    return torch.stack((c.real, c.imag), dim=1)


def two_views(z, generator):
    # Constant phase is label-preserving; a phase ramp would change distance.
    zz = z.repeat(2, 1)
    rms = zz.abs().square().mean(-1, keepdim=True).sqrt()
    noise = torch.complex(torch.randn(zz.shape, generator=generator),
                          torch.randn(zz.shape, generator=generator))
    zz = zz + .01/np.sqrt(2)*rms*noise
    phase = (torch.rand((len(zz), 1), generator=generator)*2-1)*np.pi
    return zz*torch.exp(1j*phase)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(16,16,3,stride=2,padding=1)
        self.conv2 = nn.Conv1d(16,16,3,padding=1)
        self.skip = nn.Conv1d(16,16,1,stride=2)
    def forward(self,x):
        return F.relu(self.conv2(F.relu(self.conv1(x)))+self.skip(x))


class Encoder(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(1 if kind=='power' else 2,16,7,stride=4,padding=3),
            nn.ReLU(),Block(),Block(),nn.AdaptiveAvgPool1d(16),nn.Flatten(),nn.Linear(256,64))
    def forward(self,x):
        return F.normalize(self.net(x),dim=1)


def make_head(bias=0.):
    h=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    with torch.no_grad():
        h[-1].weight.mul_(.01)
        h[-1].bias.fill_(float(bias))
    return h


def contrastive(v,y,scale,weighted=True):
    """Weighted RnC in O(M^2 log M), with exact tie handling."""
    m=len(v)
    delta=(y[:,None]-y[None,:]).abs()
    # Diagonal is excluded before sorting; sqrt floor prevents self-distance NaNs.
    sq=(v.square().sum(1)[:,None]+v.square().sum(1)[None,:]-2*v@v.T).clamp_min(1e-12)
    logits=-sq.sqrt()/TEMPERATURE
    mask=~torch.eye(m,dtype=torch.bool,device=v.device)
    d=delta[mask].view(m,m-1)
    a=logits[mask].view(m,m-1)
    ds,order=d.sort(dim=1,stable=True)
    aa=a.gather(1,order)
    weighted_logits=aa+torch.log1p(ds/scale) if weighted else aa
    suffix=torch.logcumsumexp(weighted_logits.flip(1),dim=1).flip(1)
    starts=torch.searchsorted(ds.contiguous(),ds.contiguous(),right=False)
    denom=suffix.gather(1,starts)
    return (denom-aa).mean()


def folds(meta,holdout):
    i=DISTANCES.index(holdout)
    remaining=[x for x in DISTANCES if x!=holdout]
    val=remaining[(i*2+1)%len(remaining)]
    selected=sample_indices(meta)
    tr=selected[~meta.nominal_m.iloc[selected].isin((holdout,val)).to_numpy()]
    va=selected[meta.nominal_m.iloc[selected].eq(val).to_numpy()]
    te=np.flatnonzero(meta.nominal_m.eq(holdout).to_numpy())
    for a,b in [(tr,va),(tr,te),(va,te)]:
        for column in ['file','frame_id','nominal_m']:
            assert not set(meta.iloc[a][column]) & set(meta.iloc[b][column])
    return tr,va,te,val


def balanced_validation(meta,va):
    pools=[g.index.to_numpy() for _,g in meta.iloc[va].groupby('truth_m')]
    # Fixed groups and deterministic augmentation; no changing validation target.
    batches=[]
    rng=np.random.default_rng(42011)
    for _ in range(4):
        batches.append(np.concatenate([rng.choice(p,BATCH//len(pools),replace=False) for p in pools]))
    return batches


def fit_encoder(meta,z,cell,holdout,tr,va,scale,ym,ys):
    kind,route=CELLS[cell]
    fold_i=DISTANCES.index(holdout)
    torch.manual_seed(SEED+fold_i)
    enc=Encoder(kind)
    head=make_head()
    opt=torch.optim.AdamW(list(enc.parameters())+([] if route=='rnc' else list(head.parameters())),
                          lr=.001,weight_decay=.0001)
    y=torch.tensor(meta.truth_m.to_numpy(),dtype=torch.float32)
    rng=np.random.default_rng(SEED+fold_i)
    aug=torch.Generator().manual_seed(SEED+fold_i+100)
    val_batches=balanced_validation(meta,va)
    best=np.inf;best_epoch=0;state=None;history=[];start=time.time()
    for epoch in range(1,ENC_EPOCHS+1):
        enc.train();head.train();total=0.;count=0
        order=rng.permutation(tr)
        for first in range(0,len(order),BATCH):
            ids=order[first:first+BATCH]
            raw=torch.from_numpy(np.array(z[ids],copy=True))
            xx=represent(two_views(raw,aug),kind)
            yy=y[ids].repeat(2)
            opt.zero_grad(set_to_none=True)
            v=enc(xx)
            loss=contrastive(v,yy,scale) if route=='rnc' else F.mse_loss(head(v).flatten(),(yy-ym)/ys)
            loss.backward();nn.utils.clip_grad_norm_(enc.parameters(),10)
            if route=='e2e':nn.utils.clip_grad_norm_(head.parameters(),10)
            opt.step();total+=loss.item()*len(ids);count+=len(ids)
        enc.eval();head.eval();val_losses=[]
        with torch.no_grad():
            if route=='rnc':
                vg=torch.Generator().manual_seed(71011)
                for ids in val_batches:
                    raw=torch.from_numpy(np.array(z[ids],copy=True))
                    v=enc(represent(two_views(raw,vg),kind))
                    val_losses.append(contrastive(v,y[ids].repeat(2),scale).item())
            else:
                for first in range(0,len(va),512):
                    ids=va[first:first+512]
                    v=enc(represent(torch.from_numpy(np.array(z[ids],copy=True)),kind))
                    val_losses.extend((head(v).flatten()-(y[ids]-ym)/ys).square().tolist())
        vl=float(np.mean(val_losses))
        history.append(dict(stage='encoder',epoch=epoch,train_loss=total/count,val_loss=vl))
        if vl<best-1e-6:
            best=vl;best_epoch=epoch;state=(copy.deepcopy(enc.state_dict()),copy.deepcopy(head.state_dict()))
        print('ENC',cell,holdout,epoch,'train',round(total/count,5),'val',round(vl,5),
              'best',best_epoch,'sec',round(time.time()-start,1),flush=True)
        if epoch>=16 and epoch-best_epoch>=12:break
    enc.load_state_dict(state[0]);head.load_state_dict(state[1])
    return enc,head,history,dict(encoder_best_epoch=best_epoch,encoder_epochs=epoch,
                              encoder_val_loss=best,encoder_seconds=time.time()-start)


def features(enc,z,kind,ids):
    enc.eval();vals=[]
    with torch.no_grad():
        for first in range(0,len(ids),512):
            raw=torch.from_numpy(np.array(z[ids[first:first+512]],copy=True))
            vals.append(enc(represent(raw,kind)))
    return torch.cat(vals)


def objective(pred,y,cost):
    if cost=='mse':return F.mse_loss(pred,y)
    return (int(cost)*F.relu(y-pred)+F.relu(pred-y)).mean()/(int(cost)+1)


def fit_head(vtr,vva,ytr,yva,cost,seed):
    torch.manual_seed(seed)
    bias=float(ytr.mean()) if cost=='mse' else float(torch.quantile(ytr,int(cost)/(int(cost)+1)))
    head=make_head(bias)
    opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.0001)
    best=np.inf;best_epoch=0;state=None;hist=[]
    for epoch in range(1,601):
        head.train();opt.zero_grad(set_to_none=True)
        loss=objective(head(vtr).flatten(),ytr,cost);loss.backward();opt.step()
        head.eval()
        with torch.no_grad():vl=objective(head(vva).flatten(),yva,cost).item()
        hist.append(dict(stage='head_'+str(cost),epoch=epoch,train_loss=loss.item(),val_loss=vl))
        if vl<best-1e-7:best=vl;best_epoch=epoch;state=copy.deepcopy(head.state_dict())
        if epoch>=60 and epoch-best_epoch>=80:break
    head.load_state_dict(state)
    return head,hist,dict(cost=str(cost),head_best_epoch=best_epoch,head_epochs=epoch,head_val_loss=best)


def train(cell,holdout):
    protocol()
    done=OUT/f'{cell}_holdout_{holdout}_predictions.csv.gz'
    if done.exists():print('SKIP',cell,holdout,flush=True);return
    start=time.time()
    meta=pd.read_csv(PREVIOUS/'paths.csv.gz')
    z=np.load(OUT/'raw_iq.npy',mmap_mode='r')
    tr,va,te,val=folds(meta,holdout)
    truth=meta.truth_m.to_numpy(np.float32)
    ym=float(truth[tr].mean());ys=float(truth[tr].std())
    uniq=np.unique(truth[tr]);diff=np.abs(uniq[:,None]-uniq[None,:])
    scale=float(np.median(diff[diff>0]))
    enc,base_head,history,info=fit_encoder(meta,z,cell,holdout,tr,va,scale,ym,ys)
    kind,route=CELLS[cell]
    vtr=features(enc,z,kind,tr);vva=features(enc,z,kind,va);vte=features(enc,z,kind,te)
    ytr=torch.from_numpy((truth[tr]-ym)/ys);yva=torch.from_numpy((truth[va]-ym)/ys)
    md=meta.iloc[te][['file_id','frame_id','frame_in_file','path','anchor','condition','nominal_m','truth_m','clean_los','main_m','edge40_m']].copy()
    md['sample_id']=te;md['holdout_m']=holdout;md['validation_m']=val;md['cell']=cell
    models=OUT/'models';models.mkdir(exist_ok=True)
    summaries=[]
    for cost in ('mse',)+COSTS:
        if cost=='mse' and route=='e2e':
            head=base_head;hs=dict(cost='mse',head_best_epoch=info['encoder_best_epoch'],
                                 head_epochs=info['encoder_epochs'],head_val_loss=info['encoder_val_loss'])
        else:
            head,hist,hs=fit_head(vtr,vva,ytr,yva,cost,SEED+DISTANCES.index(holdout)+1000)
            history.extend(hist)
        head.eval()
        with torch.no_grad():pred=head(vte).flatten().numpy()*ys+ym
        assert np.isfinite(pred).all()
        name='mse' if cost=='mse' else f'lambda{cost}'
        md[name+'_raw_m']=pred
        torch.save(dict(encoder=enc.state_dict(),head=head.state_dict(),cell=cell,kind=kind,
            route=route,holdout=holdout,validation_m=val,head_name=name,mean=ym,std=ys,
            weight_scale_m=scale,**info,**hs),models/f'{cell}_holdout_{holdout}_{name}.pt')
        summaries.append(dict(cell=cell,representation=kind,route=route,holdout=holdout,validation_m=val,
            train_samples=len(tr),validation_samples=len(va),test_samples=len(te),training_truths=len(uniq),
            target_mean=ym,target_std=ys,weight_scale_m=scale,
            encoder_parameters=sum(p.numel() for p in enc.parameters()),
            head_parameters=sum(p.numel() for p in head.parameters()),**info,**hs))
    pd.DataFrame(history).to_csv(models/f'{cell}_holdout_{holdout}_history.csv',index=False)
    for s in summaries:s['seconds']=time.time()-start
    pd.DataFrame(summaries).to_csv(models/f'{cell}_holdout_{holdout}_summary.csv',index=False)
    md.to_csv(done,index=False)
    print('DONE',cell,holdout,'val',val,'seconds',round(time.time()-start,1),flush=True)


def verify():
    torch.manual_seed(19)
    raw=torch.exp(-2j*np.pi*torch.arange(80)[None,:]*25/(299792458/2e6))
    p=represent(raw,'power');c=represent(raw,'complex')
    p2=torch.log10(c.square().sum(1).clamp_min(1e-6))/6
    assert torch.allclose(p[:,0],p2,atol=1e-6)
    assert abs((p.argmax().item()-NFFT//2)*(299792458/2e6)/NFFT-25)<.3
    for weighted in [False,True]:
        v=F.normalize(torch.randn(8,6),dim=1).requires_grad_()
        y=torch.tensor([1.,1.,3.,7.,1.,1.,3.,7.])
        fast=contrastive(v,y,3.,weighted)
        terms=[]
        logits=-torch.cdist(v,v)/TEMPERATURE
        for i in range(8):
            for j in range(8):
                if i==j:continue
                ids=[k for k in range(8) if k!=i and abs(y[i]-y[k])>=abs(y[i]-y[j])]
                vv=logits[i,ids]
                if weighted:vv=vv+torch.log1p((y[i]-y[ids]).abs()/3.)
                terms.append(torch.logsumexp(vv,0)-logits[i,j])
        slow=torch.stack(terms).mean()
        assert torch.allclose(fast,slow,atol=2e-5),(fast,slow)
        fast.backward();assert torch.isfinite(v.grad).all()
    for cost in COSTS:
        under=objective(torch.tensor([-1.],requires_grad=True),torch.zeros(1),cost)
        over=objective(torch.tensor([1.]),torch.zeros(1),cost)
        assert torch.isclose(under/over,torch.tensor(float(cost)))
    gen=torch.Generator().manual_seed(5)
    assert two_views(raw,gen).shape==(2,80)
    for kind in ['power','complex']:
        assert Encoder(kind)(represent(raw.repeat(3,1),kind)).shape==(3,64)
    # Hardware-only timing; no real labels or test performance used.
    e=Encoder('complex');opt=torch.optim.AdamW(e.parameters())
    r=torch.complex(torch.randn(BATCH,80),torch.randn(BATCH,80))
    y=torch.linspace(1,28,BATCH).repeat(2)
    start=time.time()
    for _ in range(5):
        opt.zero_grad();v=e(represent(two_views(r,gen),'complex'))
        contrastive(v,y,8.).backward();opt.step()
    print('VERIFIED physical grid, power/complex equivalence, weighted and unweighted RnC ties/reference/gradients, under-cost ratios, shapes',flush=True)
    print('BENCHMARK_SECONDS_PER_STEP',(time.time()-start)/5,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','verify','train'])
    ap.add_argument('--cell',choices=CELLS,nargs='+');ap.add_argument('--holdout',type=int,choices=DISTANCES)
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.mode=='prepare':prepare()
    elif args.mode=='verify':verify()
    else:
        for cell in args.cell if args.cell else CELLS:
            for holdout in [args.holdout] if args.holdout else DISTANCES:train(cell,holdout)
