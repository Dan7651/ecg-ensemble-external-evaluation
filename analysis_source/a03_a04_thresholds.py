"""Deposited A03/A04 saved-score threshold diagnostics, with full bootstrap refits.

No neural model, waveform, new model selection or deployment threshold update.
Fixed score histograms avoid repeating sorts; selection exactly matches the
deposited oracle_f1/fixed_specificity_threshold functions including tie rules.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import fcntl
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
from datetime import datetime,timezone
import numpy as np
import scipy
import sklearn
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/a03_a04_thresholds'
ARMS=['resnet','jepa','blend'];CLASSES=['NORM','MI','STTC','CD','HYP']
CRITERIA=['f1','specificity_0_90','specificity_0_95','specificity_0_99']
METRICS=['f1','sensitivity','specificity','false_positive_rate','precision']
TRANSPORT=np.array([.48,.37,.42,.43,.20],float)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
read=lambda f:json.loads(Path(f).read_text())


def plain(v):
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    return v


def atomic(path,value,binary=False,immutable=False):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    if immutable and path.exists():
        assert not binary and read(path)==plain(value),'Preserve differing immutable artifact';return
    temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f:
        if binary:np.savez_compressed(f,**value)
        else:f.write((json.dumps(plain(value),indent=2,allow_nan=False)+'\n').encode())
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def confusion_metrics(counts):
    """counts[...,4]=TP,FP,FN,TN; missing support propagates explicitly."""
    x=np.asarray(counts,float);tp,fp,fn,tn=np.moveaxis(x,-1,0)
    div=lambda a,b,zero=0.:np.divide(a,b,out=np.full_like(np.asarray(a,float),zero),where=b>0)
    result=np.stack([div(2*tp,2*tp+fp+fn),div(tp,tp+fn,np.nan),
        div(tn,tn+fp,np.nan),div(fp,tn+fp,np.nan),div(tp,tp+fp)],axis=-1)
    return np.where(np.isfinite(x).all(-1)[...,None],result,np.nan)


class Column:
    def __init__(self,y,p,folds):
        self.y=np.asarray(y,dtype=np.uint8);self.p=np.asarray(p,float);self.folds=np.asarray(folds,int)
        assert self.y.ndim==self.p.ndim==1 and self.y.shape==self.p.shape==self.folds.shape
        assert np.isin(self.y,[0,1]).all() and np.isfinite(self.p).all() and ((self.p>=0)&(self.p<=1)).all()
        assert np.isin(self.folds,range(5)).all()
        self.unique,self.bins=np.unique(self.p,return_inverse=True);self.m=len(self.unique)
        self.fold_bins=self.folds*self.m+self.bins
        self.candidates=[];self.cutoffs=[]
        for held in [-1,0,1,2,3,4]:
            mask=np.ones(len(y),bool) if held==-1 else self.folds!=held
            t=np.unique(np.r_[self.p[mask],0.,np.nextafter(1.,np.inf)])
            self.candidates.append(t);self.cutoffs.append(np.searchsorted(self.unique,t,side='left'))

    def histograms(self,w):
        w=np.asarray(w,float)
        assert w.shape==self.y.shape and np.isfinite(w).all() and (w>=0).all() and w.sum()>0
        hp=np.bincount(self.fold_bins,weights=w*self.y,minlength=5*self.m).reshape(5,self.m)
        hn=np.bincount(self.fold_bins,weights=w*(1-self.y),minlength=5*self.m).reshape(5,self.m)
        gp,gn=hp.sum(0),hn.sum(0)
        fp=np.vstack([gp,gp[None]-hp]);fn=np.vstack([gn,gn[None]-hn])
        self.train_pos=fp.sum(1);self.train_neg=fn.sum(1)
        self.tp=np.c_[np.cumsum(fp[:,::-1],axis=1)[:,::-1],np.zeros(6)]
        self.fp=np.c_[np.cumsum(fn[:,::-1],axis=1)[:,::-1],np.zeros(6)]
        self.test_tp=np.c_[np.cumsum(hp[:,::-1],axis=1)[:,::-1],np.zeros(5)]
        self.test_fp=np.c_[np.cumsum(hn[:,::-1],axis=1)[:,::-1],np.zeros(5)]
        self.test_pos=hp.sum(1);self.test_neg=hn.sum(1)

    def evaluate(self,threshold,held=-1):
        if not np.isfinite(threshold):return np.full(4,np.nan)
        k=np.searchsorted(self.unique,threshold,side='left')
        if held==-1:
            tp,fp=self.tp[0,k],self.fp[0,k];pos,neg=self.train_pos[0],self.train_neg[0]
        else:
            tp,fp=self.test_tp[held,k],self.test_fp[held,k];pos,neg=self.test_pos[held],self.test_neg[held]
        return np.array([tp,fp,pos-tp,neg-fp])

    def fits(self):
        thresholds=np.full((6,4),np.nan)
        for k in range(6):
            if self.train_pos[k]<=0 or self.train_neg[k]<=0:continue
            cuts=self.cutoffs[k];t=self.candidates[k];tp=self.tp[k,cuts];fp=self.fp[k,cuts]
            fn=self.train_pos[k]-tp;tn=self.train_neg[k]-fp
            f=2*tp/(2*tp+fp+fn);best=np.flatnonzero(f==f.max())
            ix=best[np.argmin(abs(t[best]-.5))];thresholds[k,0]=t[ix]
            spec=tn/(tn+fp)
            for c,target in enumerate([.9,.95,.99],1):
                eligible=np.flatnonzero(spec>=target)
                best=eligible[tp[eligible]==tp[eligible].max()]
                thresholds[k,c]=t[best[0]]
        return thresholds


def configurations():
    out=[]
    for arm in ARMS:
        for kind in ['fixed_0_5','external_oracle_f1','patient_crossfit_f1']:
            out.append(dict(arm=arm,kind=kind))
        for target in [.9,.95,.99]:
            for kind in ['external_oracle_specificity','patient_crossfit_specificity']:
                out.append(dict(arm=arm,kind=kind,target_specificity=target))
    out.extend([dict(arm='blend',kind='transported_development'),dict(arm='blend',kind='retrospective_prevalence_adjusted')])
    assert len(out)==29
    return out


def prepare_columns(y,probabilities,folds):
    return [[Column(y[:,c],p[:,c],folds) for c in range(5)] for p in probabilities]


def one_draw(columns,w,source_prevalence):
    fits=np.empty((3,6,4,5));config=configurations();counts=np.full((29,5,4),np.nan)
    prevalence=np.array([np.dot(w,columns[0][c].y)/w.sum() for c in range(5)])
    if ((prevalence>0)&(prevalence<1)).all():
        delta=logit(prevalence)-logit(source_prevalence);adjusted=expit(logit(TRANSPORT)-delta)
    else:adjusted=np.full(5,np.nan)
    for a in range(3):
        for c,col in enumerate(columns[a]):
            col.histograms(w);t=col.fits();fits[a,:,:,c]=t
            for row,spec in enumerate(config):
                if ARMS[a]!=spec['arm']:continue
                kind=spec['kind']
                if kind=='fixed_0_5':counts[row,c]=col.evaluate(.5)
                elif kind=='transported_development':counts[row,c]=col.evaluate(TRANSPORT[c])
                elif kind=='retrospective_prevalence_adjusted':counts[row,c]=col.evaluate(adjusted[c])
                elif kind=='external_oracle_f1':counts[row,c]=col.evaluate(t[0,0])
                elif kind=='external_oracle_specificity':
                    criterion=1+[.9,.95,.99].index(spec['target_specificity']);counts[row,c]=col.evaluate(t[0,criterion])
                else:
                    criterion=0 if kind=='patient_crossfit_f1' else 1+[.9,.95,.99].index(spec['target_specificity'])
                    if np.isfinite(t[1:,criterion]).all():
                        counts[row,c]=sum((col.evaluate(t[f+1,criterion],f) for f in range(5)),np.zeros(4))
    return confusion_metrics(counts),fits,adjusted,prevalence,counts


def checkpoint(arrays,meta):
    path=OUT/f'progress_{(meta["completed"]//50)%2}.npz'
    atomic(path,dict(**arrays,_meta=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8)),True)
    atomic(path.with_suffix('.json'),dict(completed=meta['completed'],sha256=sha(path)))


def resume(lock):
    states=[];existing=False
    for n in range(2):
        path=OUT/f'progress_{n}.npz';side=path.with_suffix('.json');existing|=path.exists() or side.exists()
        if not path.exists() or not side.exists():continue
        try:
            assert sha(path)==read(side)['sha256']
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(z['_meta'].tobytes());arrays={k:z[k].copy() for k in z.files if k!='_meta'}
            if meta['execution_plan_sha256']!=lock:raise RuntimeError('Saved scientific plan changed')
            states.append((meta,arrays))
        except (OSError,ValueError,KeyError,AssertionError):continue
    if existing and not states:raise RuntimeError('No intact saved generation; preserve and diagnose')
    return max(states,key=lambda x:x[0]['completed']) if states else None


def summarize(point_values,draws,config):
    sys.path.insert(0,str(ROOT));from analysis_core import interval
    result=[];bases={a:next(i for i,s in enumerate(config) if s==dict(arm=a,kind='fixed_0_5')) for a in ARMS}
    for k,spec in enumerate(config):
        delta=draws[:,k]-draws[:,bases[spec['arm']]]
        result.append(dict(**spec,per_class=point_values[k],macro=point_values[k].mean(0),
            difference_from_same_arm_fixed_0_5=point_values[k]-point_values[bases[spec['arm']]],
            conditional_patient_intervals=dict(per_class=[[interval(draws[:,k,c,m]) for m in range(5)] for c in range(5)],
                macro=[interval(draws[:,k,:,m].mean(1)) for m in range(5)],
                difference_per_class=[[interval(delta[:,c,m]) for m in range(5)] for c in range(5)],
                difference_macro=[interval(delta[:,:,m].mean(1)) for m in range(5)])))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);mutex=(OUT/'analysis.lock').open('a+');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from public_protocol_transport_v3 import require_live_deposit
    deposit=require_live_deposit(ROOT)
    atomic(OUT/f'public_deposit_refresh_{time.time_ns()}.json',deposit,immutable=True)
    assert (np.__version__,scipy.__version__,sklearn.__version__)==('1.26.4','1.12.0','1.5.2')
    if (OUT/'assessment.json').exists():
        done=read(OUT/'assessment.json');assert done['source_sha256']==sha(__file__) and done['draws_sha256']==sha(OUT/'patient_draws.npz')
        print('A03_A04_COMPLETED_ASSESSMENT_REUSED_NO_RERUN',flush=True);return
    sys.path.insert(0,str(ROOT));from a05_calibration_offsets import load_inputs
    from analysis_core import patient_folds,prevalence_adjustment
    sy,sp,_,y,patients,probs,pins=load_inputs();source_prevalence=sy.sum(0,dtype=np.float64)/len(sy);folds=patient_folds(patients)
    unique,inv=np.unique(patients,return_inverse=True)
    assert len(unique)==20401 and len(patients)==21183 and len(set(sp))==1854
    assert all(not(set(patients[folds==f])&set(patients[folds!=f])) for f in range(5))
    plan=dict(deposit_doi=deposit['doi'],deposited_protocol_sha256=deposit['protocol_sha256'],source_sha256=sha(__file__),
        analysis_core_sha256=sha(ROOT/'analysis_core.py'),input_loader_sha256=sha(ROOT/'a05_calibration_offsets.py'),
        public_transport_sha256=sha(ROOT/'public_protocol_transport_v3.py'),public_transport_base_sha256=sha(ROOT/'coverage_code_v2/public_protocol_retry.py'),inputs=pins,
        metrics=METRICS,class_order=CLASSES,configuration_order=configurations(),training_models_unchanged=True,
        records=21183,patients=20401,source_records=2129,source_patients=1854,source_prevalence_fixed=source_prevalence,
        threshold_vector=TRANSPORT,specificity_targets=[.9,.95,.99],draws=2000,seed=2026102701,minimum_valid=1960,
        fold_seed=2026102703,fold_records=[int((folds==f).sum()) for f in range(5)],fold_patients=[len(set(patients[folds==f])) for f in range(5)],
        threshold_axis_order=['all_data','exclude_fold_0','exclude_fold_1','exclude_fold_2','exclude_fold_3','exclude_fold_4'],criteria=CRITERIA,
        uncertainty='External patient resampling conditional on observed source prevalence; refit every oracle/crossfit/prior-adjusted threshold in each draw; fixed patient fold assignment',
        candidate_grid='Original fit-partition score values plus0andnextafter1, including zero-frequency rows, exactly as deposited weighted functions',
        aggregation='Pool heldout confusion counts; never average fold F1',environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__))
    atomic(OUT/'execution_plan.json',plan,immutable=True);lock=sha(OUT/'execution_plan.json')
    from sph_component_analysis_v1 import mapping_and_rows
    _,rows=mapping_and_rows(ROOT.parents[1]/'experiments/sph_component_manuscript_20261026_v1')
    ids=np.array([r['id'] for r in rows]);assert len(set(ids))==21183
    assert patients.tolist()==[r['patient_id'] for r in rows] and np.array_equal(y,[r['labels'] for r in rows])
    identities=OUT/'population.npz'
    if identities.exists():
        with np.load(identities,allow_pickle=False) as z:assert np.array_equal(z['ids'],ids) and np.array_equal(z['patients'],patients) and np.array_equal(z['fold'],folds) and np.array_equal(z['y'],y)
    else:atomic(identities,dict(ids=ids,patients=patients,fold=folds,y=y),True)
    columns=prepare_columns(y,probs,folds);values,thresholds,adjusted,prev,counts=one_draw(columns,np.ones(len(y)),source_prevalence)
    delta,expected_adjusted=prevalence_adjustment(source_prevalence,prev,TRANSPORT);np.testing.assert_array_equal(adjusted,expected_adjusted)
    atomic(OUT/'point_estimates.json',dict(per_class_metrics=values,pooled_confusion_counts=counts,thresholds=thresholds,adjusted_thresholds=adjusted,source_prevalence=source_prevalence,external_prevalence=prev,prior_log_odds_delta=delta),immutable=True)
    rng=np.random.default_rng(2026102701);start=0;d=np.full((2000,29,5,5),np.nan);t=np.full((2000,3,6,4,5),np.nan);pt=np.full((2000,5),np.nan);pe=np.full((2000,5),np.nan)
    previous=resume(lock)
    if previous:
        meta,arrays=previous;start=meta['completed'];d[:start]=arrays['metrics'];t[:start]=arrays['thresholds'];pt[:start]=arrays['prior_thresholds'];pe[:start]=arrays['prevalence'];rng.bit_generator.state=meta['rng']
    started=time.monotonic()
    for b in range(start,2000):
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv]
        d[b],t[b],pt[b],pe[b],_=one_draw(columns,w,source_prevalence)
        if (b+1)%50==0:
            checkpoint(dict(metrics=d[:b+1],thresholds=t[:b+1],prior_thresholds=pt[:b+1],prevalence=pe[:b+1]),dict(completed=b+1,execution_plan_sha256=lock,rng=rng.bit_generator.state))
            atomic(OUT/'status.json',dict(phase='full_threshold_refit_patient_bootstrap',completed=b+1,planned=2000))
            print('A03_A04_PATIENT_DRAWS',b+1,2000,flush=True)
    atomic(OUT/'patient_draws.npz',dict(per_class_metrics=d,thresholds=t,prior_thresholds=pt,external_prevalence=pe),True)
    result=dict(status='A03_A04_COMPLETED_REQUIRES_INDEPENDENT_VERIFICATION',source_sha256=sha(__file__),execution_plan_sha256=lock,
        point_sha256=sha(OUT/'point_estimates.json'),population_sha256=sha(identities),draws_sha256=sha(OUT/'patient_draws.npz'),
        metrics=METRICS,class_order=CLASSES,configurations=summarize(values,d,configurations()),draws=2000,seed=2026102701,minimum_valid=1960,
        point_thresholds=thresholds,source_prevalence=source_prevalence,external_prevalence=prev,prior_log_odds_delta=delta,adjusted_thresholds=adjusted,
        threshold_intervals=[[[[__import__('analysis_core').interval(t[:,a,k,c,cl]) for cl in range(5)] for c in range(4)] for k in range(6)] for a in range(3)],
        prior_threshold_intervals=[__import__('analysis_core').interval(pt[:,cl]) for cl in range(5)],
        bootstrap_seconds_this_execution=time.monotonic()-started,
        limits=['Original historically viewed selected SPH cohort; not fresh confirmation.',
            'External oracle fits and evaluates on the same cohort; optimistic retrospective upper bound, not transfer performance.',
            'Patient crossfit is internal threshold validation in the target cohort, not independent cohort transfer.',
            'Prior adjustment uses observed target prevalence; conditional on fixed observed development prevalence and prior-shift assumptions.',
            'No stored probability, model, mapping or deployment threshold is updated.',
            'Many pointwise secondary intervals; no selection-adjusted or clinical-safety/benchmark-win claim.'])
    atomic(OUT/'assessment.json',result,immutable=True);atomic(OUT/'status.json',dict(phase='completed_requires_independent_verification',assessment_sha256=sha(OUT/'assessment.json')))
    print('A03_A04_COMPLETED',sha(OUT/'assessment.json'),flush=True)


if __name__=='__main__':main()
