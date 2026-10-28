"""Post hoc M1/M2/M5/M6 diagnostics from frozen saved scores; no model runs."""
from pathlib import Path
import csv,gzip,hashlib,json,sys,itertools
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score

ROOT=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--metadata',type=Path,required=True);parser.add_argument('--codebook',type=Path,required=True);parser.add_argument('--crossed',action='store_true');args=parser.parse_args()
assert hashlib.sha256(args.metadata.read_bytes()).hexdigest()=='c6f8dc5197758a3970aae554f8b6ee884d96656b121879bc0eb1adf367ea6d86'
P=ROOT/'reproduced_followup';P.mkdir(exist_ok=True);(P/'analysis').mkdir(exist_ok=True)
CL=['NORM','MI','STTC','CD','HYP'];SEEDS=[101,202,303,404,505]
from saved_score_statistics import ScoreStatistics

def dump(name,value):
    (P/'analysis'/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(name,rows):
    with (P/'analysis'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric(y,p):
    return np.array([roc_auc_score(y,p,average='macro'),average_precision_score(y,p,average='macro'),
        f1_score(y,p>=.5,average='macro',zero_division=0),((p.astype(float)-y)**2).mean()])
def pairlist(cyclic=False):
    return [(group,i,j,i if group!='JJ' else i+5,j if group=='RR' else j+5)
        for group in ['RR','JJ','RJ'] for i in range(5) for j in range(5)
        if ((j==(i+1)%5) if cyclic else (group=='RJ' or i<j))]
def pair_gain(a,b,m):
    out=m-np.maximum(a,b);out[3]=min(a[3],b[3])-m[3];return out
def safe_corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None

def load():
    metadata={r['ECG_ID']:r for r in csv.DictReader(args.metadata.open())}
    mapping=json.loads((ROOT/'data/frozen_primary_code_map.json').read_text())['groups']
    with gzip.open(ROOT/'scores/matched_seed_selected_21183.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    ids=[r['ECG_ID'] for r in rows]
    y=np.array([[bool(set(int(t.split('+')[0]) for t in metadata[r['ECG_ID']]['AHA_Code'].split(';'))&set(mapping[c])) for c in CL] for r in rows],np.int8)
    scores=[np.array([[r[f'{fam}{seed}_{c}'] for c in CL] for r in rows],np.float32) for fam in ['R','J'] for seed in SEEDS]
    patients=np.array([metadata[i]['Patient_ID'] for i in ids])
    assert y.shape==(21183,5) and len(set(patients))==20401
    return y,scores,ids,patients,metadata

def saved_score_diagnostics():
    y,scores,ids,patients,metadata=load();single=np.array([metric(y,p) for p in scores])
    rows=[];oracle_rows=[];gain_rows=[];diversity_summary=[]
    for group,i,j,a,b in pairlist():
        pa,pb=scores[a],scores[b];ea=(pa>=.5)!=y;eb=(pb>=.5)!=y
        correlations=[];phis=[];disagreements=[]
        for c,target in enumerate(CL):
            cor=safe_corr(pa[:,c],pb[:,c]);phi=safe_corr(ea[:,c],eb[:,c]);dis=float(np.mean((pa[:,c]>=.5)!=(pb[:,c]>=.5)))
            correlations.append(cor);phis.append(phi);disagreements.append(dis)
            rows.append(dict(family=group,seed_a=SEEDS[i],seed_b=SEEDS[j],target=target,score_pearson=cor,error_phi=phi,decision_disagreement=dis))
        rows.append(dict(family=group,seed_a=SEEDS[i],seed_b=SEEDS[j],target='Macro',score_pearson=float(np.mean(correlations)),error_phi=float(np.mean(phis)),decision_disagreement=float(np.mean(disagreements))))
        blend=(pa+pb)*.5;m=metric(y,blend);g=pair_gain(single[a],single[b],m)
        gain_rows.append(dict(family=group,i=i,j=j,metrics=m.tolist(),gain=g.tolist()))
        selector=np.where(y==1,np.maximum(pa,pb),np.minimum(pa,pb));om=metric(y,selector)
        oracle_rows.append(dict(family=group,seed_a=SEEDS[i],seed_b=SEEDS[j],**{f'oracle_{key}':v for key,v in zip(['auroc','ap','f1','brier'],om)},**{f'blend_{key}':v for key,v in zip(['auroc','ap','f1','brier'],m)}))
    table('all_pair_diversity.csv',rows);table('oracle_selector_bound.csv',oracle_rows);dump('all_pair_metrics.json',gain_rows)
    for scope in ['all45','cyclic15']:
        for group in ['RR','JJ','RJ']:
            for target in CL+['Macro']:
                take=[r for r in rows if r['family']==group and r['target']==target and (scope=='all45' or r['seed_b']==SEEDS[(SEEDS.index(r['seed_a'])+1)%5] or (group!='RJ' and r['seed_a']==SEEDS[0] and r['seed_b']==SEEDS[4]))]
                # The wraparound within-family edge is stored once as (101,505).
                for key in ['score_pearson','error_phi','decision_disagreement']:
                    values=np.asarray([r[key] for r in take],float)
                    diversity_summary.append(dict(scope=scope,family=group,target=target,metric=key,pairs=len(values),mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),minimum=float(values.min()),maximum=float(values.max())))
    table('diversity_summary.csv',diversity_summary)
    loo=[]
    metrics_map={(r['family'],min(r['i'],r['j']) if r['family']!='RJ' else r['i'],max(r['i'],r['j']) if r['family']!='RJ' else r['j']):np.array(r['gain']) for r in gain_rows}
    for scope in ['cyclic15','all45']:
        for family in ['none','R','J']:
            for omitted in ([None] if family=='none' else range(5)):
                values={g:[] for g in ['RR','JJ','RJ']}
                for group,i,j,a,b in pairlist(scope=='cyclic15'):
                    if family=='R' and (a==omitted or b==omitted):continue
                    if family=='J' and (a==5+omitted or b==5+omitted):continue
                    key=(group,min(i,j),max(i,j)) if group!='RJ' else (group,i,j)
                    values[group].append(metrics_map[key])
                means={g:np.mean(v,axis=0) for g,v in values.items()}
                loo.append(dict(scope=scope,omitted_family=family,omitted_seed='' if omitted is None else SEEDS[omitted],RR_pairs=len(values['RR']),JJ_pairs=len(values['JJ']),RJ_pairs=len(values['RJ']),RR_auroc_gain=means['RR'][0],JJ_auroc_gain=means['JJ'][0],RJ_auroc_gain=means['RJ'][0],RJ_minus_RR=means['RJ'][0]-means['RR'][0],RJ_minus_JJ=means['RJ'][0]-means['JJ'][0],RJ_ap_gain=means['RJ'][1],RJ_f1_gain=means['RJ'][2],RJ_brier_gain=means['RJ'][3]))
    table('leave_one_seed_out.csv',loo)
    # Inclusion: unchanged original illustrative probabilities on all25770.
    scorepath=ROOT/'scores/original_first_window_all_25770.csv.gz'
    with gzip.open(scorepath,'rt') as f:allrows=list(csv.DictReader(f))
    excluded=[r for r in allrows if r['selected']=='0'];assert len(excluded)==4587
    statements=[set(int(t.split('+')[0]) for t in metadata[r['ECG_ID']]['AHA_Code'].split(';')) for r in excluded]
    codebook={int(r['Code']):r for r in csv.DictReader(args.codebook.open())}
    arms={a:np.array([float(r[f'{p}_NORM']) for r in excluded],np.float32) for a,p in [('resnet','R42'),('jepa','J42')]};arms['blend']=(arms['resnet']+arms['jepa'])*.5
    rates=[]
    groups=[('AF',np.array([50 in s for s in statements])),('Atrial flutter',np.array([51 in s for s in statements])),('AF or flutter',np.array([bool(s&{50,51}) for s in statements])),('Only sinus-node statements',np.array([bool(s) and s<={21,22,23} for s in statements])),('Other non-AF/flutter statements',np.array([not(s&{50,51}) and not(s<={21,22,23}) for s in statements]))]
    for code in sorted(set().union(*statements)):
        groups.append((str(code)+': '+codebook[code]['Description'],np.array([code in s for s in statements])))
    for group,mask in groups:
        for arm,prob in arms.items():
            n=int(mask.sum());positive=int((prob[mask]>=.5).sum())
            rates.append(dict(statement_group=group,arm=arm,records=n,NORM_positive=positive,NORM_positive_rate=positive/n))
    table('excluded_norm_statement_rates.csv',rates)
    assert next(r['records'] for r in rates if r['statement_group']=='AF')==285
    assert next(r['records'] for r in rates if r['statement_group']=='Atrial flutter')==7
    # Prevalence sensitivity changes only assumed priors, never score values.
    a05=json.loads((ROOT/'data/followup_source_prevalence.json').read_text());sourceprev=np.array(a05['source_prevalence'])
    selected=[r for r in allrows if r['selected']=='1'];assert [r['ECG_ID'] for r in selected]==ids
    pr=np.array([[r[f'R42_{c}'] for c in CL] for r in selected],np.float32);pj=np.array([[r[f'J42_{c}'] for c in CL] for r in selected],np.float32);blend=(pr+pj)*.5
    factors=json.loads((ROOT/'FOLLOWUP_ANALYSIS_PLAN.json').read_text())['M5']['prior_factors'];sensitivity=[]
    for factor in factors:
        assumed=y.mean(0)*factor;valid=(assumed>0)&(assumed<1);threshold=np.full(5,np.nan)
        threshold[valid]=expit(logit([.48,.37,.42,.43,.20])[valid]-(logit(assumed[valid])-logit(sourceprev[valid])))
        fs=[]
        for c,target in enumerate(CL):
            val=float(f1_score(y[:,c],blend[:,c]>=threshold[c],zero_division=0)) if valid[c] else None
            fs.append(val);sensitivity.append(dict(factor=factor,target=target,assumed_prevalence=float(assumed[c]),valid=bool(valid[c]),threshold=float(threshold[c]) if valid[c] else '',f1=val if valid[c] else ''))
        sensitivity.append(dict(factor=factor,target='All5 macro',assumed_prevalence='',valid=bool(valid.all()),threshold='',f1=float(np.mean(fs)) if valid.all() else ''))
        sensitivity.append(dict(factor=factor,target='Abnormal4 macro',assumed_prevalence='',valid=bool(valid[1:].all()),threshold='',f1=float(np.mean(fs[1:])) if valid[1:].all() else ''))
    table('prior_misspecification.csv',sensitivity)
    dump('saved_score_completion.json',dict(completed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((ROOT/'FOLLOWUP_ANALYSIS_PLAN.json').read_bytes()).hexdigest(),new_neural_execution=False,pairs=45,oracle='Label-dependent nondeployable upper bound',single_metrics=single.tolist(),prior_factor1_macro=next(r['f1'] for r in sensitivity if r['factor']==1 and r['target']=='All5 macro')))
    print('M1_M2_M5_LEAVE_ONE_OUT_COMPLETED',flush=True)

def crossed():
    y,scores,ids,patients,metadata=load();plan=json.loads((ROOT/'FOLLOWUP_ANALYSIS_PLAN.json').read_text())['M6']['crossed']
    rng=np.random.default_rng(plan['seed']);_,inv=np.unique(patients,return_inverse=True);n=inv.max()+1
    single_stats=[ScoreStatistics(y,p) for p in scores]
    pair_stats={(a,b):ScoreStatistics(y,(scores[a]+scores[b])*.5) for a in range(10) for b in range(a+1,10)}
    progress=P/'analysis/crossed_progress.npz';start=0;draws=np.full((plan['draws'],3,4),np.nan)
    if progress.exists():
        with np.load(progress,allow_pickle=False) as z:
            meta=json.loads(z['metadata'].tobytes());start=meta['completed'];draws[:start]=z['draws'];rng.bit_generator.state=meta['rng']
            assert meta['plan_sha256']==hashlib.sha256((ROOT/'FOLLOWUP_ANALYSIS_PLAN.json').read_bytes()).hexdigest()
    for b in range(start,plan['draws']):
        sampled=[rng.integers(5,size=5),rng.integers(5,size=5)+5]
        w=np.bincount(rng.integers(n,size=n),minlength=n)[inv]
        chosen=set(np.concatenate(sampled).tolist());single={i:single_stats[i].evaluate(w)['macro'][[0,1,2,6]] for i in chosen};cache={}
        def gain(a,c):
            if a==c:return np.zeros(4)
            key=tuple(sorted([a,c]))
            if key not in cache:cache[key]=pair_gain(single[a],single[c],pair_stats[key].evaluate(w)['macro'][[0,1,2,6]])
            return cache[key]
        rr=np.mean([gain(a,c) for a,c in itertools.combinations(sampled[0],2)],axis=0)
        jj=np.mean([gain(a,c) for a,c in itertools.combinations(sampled[1],2)],axis=0)
        rj=np.mean([gain(a,c) for a,c in itertools.product(*sampled)],axis=0)
        draws[b]=[rj-rr,rj-jj,rj-(rr+jj)/2]
        if (b+1)%50==0:
            meta=dict(completed=b+1,rng=rng.bit_generator.state,plan_sha256=hashlib.sha256((ROOT/'FOLLOWUP_ANALYSIS_PLAN.json').read_bytes()).hexdigest())
            tmp=progress.with_suffix('.partial.npz');np.savez_compressed(tmp,draws=draws[:b+1],metadata=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8));tmp.replace(progress)
            print('CROSSED_MODEL_PATIENT_DRAWS',b+1,plan['draws'],flush=True)
    valid=np.isfinite(draws).all((1,2));assert valid.sum()>=980
    np.savez_compressed(P/'analysis/crossed_model_patient_draws.npz',contrasts=draws)
    dump('crossed_model_patient.json',dict(draws=plan['draws'],valid=int(valid.sum()),seed=plan['seed'],contrast_order=['RJ_minus_RR','RJ_minus_JJ','RJ_minus_mean_within'],metrics=['auroc','ap','f1','brier'],ci95=np.percentile(draws[valid],[2.5,97.5],axis=0).tolist(),median=np.median(draws[valid],axis=0).tolist(),scope=plan['interpretation'],identical_fit_duplicate_edges_zero=True))
    print('CROSSED_MODEL_PATIENT_COMPLETED',flush=True)

if __name__=='__main__':
    saved_score_diagnostics()
    if args.crossed:crossed()
