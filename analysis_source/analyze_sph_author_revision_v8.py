"""Additional post hoc saved-score analyses. No waveform/model execution."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import collections,csv,datetime,hashlib,itertools,json,sys,time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score

H=Path(__file__).resolve().parent
P=H/'experiments/sph_prospective_extension_20261027_v1'
V=H/'experiments/sph_author_revision_20261028_v8'
O=V/'analysis';O.mkdir(exist_ok=True)
sys.path[:0]=[str(H),str(P)]
from sph_component_analysis_v1 import mapping_and_rows,load_predictions
from sph_component_statistics_v1 import ScoreStatistics
CL=['NORM','MI','STTC','CD','HYP'];SEEDS=[101,202,303,404,505]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x

def save(name,x):
    (O/name).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')

def csvsave(name,rows):
    assert rows
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))

def ci(x,minimum=1960):
    x=np.asarray(x);g=np.isfinite(x)
    return {'valid':int(g.sum()),'ci95':np.percentile(x[g],[2.5,97.5]).tolist() if g.sum()>=minimum else None}

def main():
    assert read(V/'plan_receipt.json')['analysis_plan_sha256']==sha(V/'analysis_plan.json')
    assert not (O/'completion.json').exists(),'Completed revision exists; preserve it'
    start=time.monotonic()
    a=read(P/'observed_a01_followup_completed_v1/control_assessment.json')
    assert sha(P/'observed_a01_followup_completed_v1/control_assessment.json')=='ddb20dc35ef4f28bb4814b3e2964bf7e0743ef426722dd3658db6ca63c09dd39'
    m=np.asarray(a['singles']);g=np.asarray(a['primary_gains']);ens=np.asarray(a['primary_ensembles']).reshape(3,5,4)
    component=[]
    for fam,values in [('ResNet',m[:5]),('JEPA head',m[5:])]:
        for j,metric in enumerate(a['metrics']):
            z=values[:,j];component.append(dict(family=fam,metric=metric,mean=z.mean(),sample_sd=z.std(ddof=1),minimum=z.min(),maximum=z.max(),fits=5))
    csvsave('component_seed_summary.csv',component)
    fp=P/'observed_a01_followup_completed_v1/patient_bootstrap_draws.npz';assert sha(fp)==a['draws_sha256']
    with np.load(fp,allow_pickle=False) as z:gd=z['gain'].copy();md=z['macro'].copy()
    sensitivities=[]
    for mode in ['all_five','omit_ResNet202_only','omit_both_seed202']:
        for f,fam in enumerate(['RR','JJ','RJ']):
            keep=[]
            for i in range(5):
                pair=a['primary_pair_definitions'][f*5+i];ii,jj=pair[1:]
                identities=[('R' if fam!='JJ' else 'J',SEEDS[ii]),('R' if fam=='RR' else 'J',SEEDS[jj])]
                exclude=mode=='omit_ResNet202_only' and ('R',202) in identities
                exclude |= mode=='omit_both_seed202' and any(s==202 for _,s in identities)
                if not exclude:keep.append(i)
            for j,metric in enumerate(a['metrics']):
                vals=g[f,keep,j];drawmean=gd[:,f,keep,j].mean(1)
                sensitivities.append(dict(sensitivity=mode,family=fam,metric=metric,blocks=len(keep),ensemble_mean=ens[f,keep,j].mean(),
                    gain_mean=vals.mean(),gain_sd=vals.std(ddof=1),gain_min=vals.min(),gain_max=vals.max(),positive_blocks=int((vals>0).sum()),
                    patient_ci_low=ci(drawmean)['ci95'][0],patient_ci_high=ci(drawmean)['ci95'][1]))
    csvsave('seed202_sensitivity.csv',sensitivities)
    # Whole-model resampling preserves the dependence of pair edges.
    mat=np.zeros((10,10,4));known=set()
    for pair in a['all_45_sensitivity_pairs']:
        i=SEEDS.index(pair['seed_a'])+(5 if pair['family']=='JJ' else 0)
        j=SEEDS.index(pair['seed_b'])+(0 if pair['family']=='RR' else 5)
        assert i!=j;mat[i,j]=mat[j,i]=pair['gain'];known.add(tuple(sorted([i,j])))
    assert len(known)==45
    upper=list(itertools.combinations(range(5),2))
    def family_stat(rr,jj):
        withinR=np.mean([mat[rr[i],rr[j]] for i,j in upper],0)
        withinJ=np.mean([mat[jj[i],jj[j]] for i,j in upper],0)
        cross=mat[np.ix_(rr,jj)].mean((0,1))
        return np.stack([withinR,withinJ,cross,cross-withinR,cross-withinJ,cross-(withinR+withinJ)/2])
    observed=family_stat(np.arange(5),np.arange(5,10));rng=np.random.default_rng(2026102802)
    modeldraw=np.stack([family_stat(rng.integers(0,5,5),rng.integers(5,10,5)) for _ in range(10000)])
    perms=[]
    for group in itertools.combinations(range(10),5):
        complement=[i for i in range(10) if i not in group];perms.append(family_stat(np.array(group),np.array(complement))[-1])
    perms=np.asarray(perms)
    model={'metrics':a['metrics'],'rows':['RR','JJ','RJ','RJ_minus_RR','RJ_minus_JJ','RJ_minus_average_same_family'],
      'all_pair_observed':observed,'model_seed_bootstrap_ci95':np.percentile(modeldraw,[2.5,97.5],axis=0),
      'model_seed_draws':10000,'model_seed_bootstrap_seed':2026102802,
      'exact_model_label_assignments':252,'permutation_statistic':observed[-1],
      'permutation_upper_tail_fraction':np.mean(perms>=observed[-1]-1e-15,axis=0),
      'limitation':'Target scores and shared encoder fixed. Model-label exchangeability is not established; tail fractions are conditional diagnostics, not randomized or causal p-values. Pair edges are never treated as independent fits.'}
    save('model_seed_uncertainty.json',model)
    np.savez_compressed(O/'model_resampling_draws.npz',model_seed_bootstrap=modeldraw,model_label_permutations=perms)
    # Paired rule-dependent F1 and predictive values reuse the very same patient draws.
    t=read(P/'results/a03_a04_thresholds/assessment.json');pts=read(P/'results/a03_a04_thresholds/point_estimates.json')
    tf=P/'results/a03_a04_thresholds/patient_draws.npz';assert sha(tf)==t['draws_sha256']
    with np.load(tf,allow_pickle=False) as z:td=z['per_class_metrics'].copy();prev=z['external_prevalence'].copy()
    cfg=t['configurations'];pairs=[]
    for kind in ['fixed_0_5','external_oracle_f1','patient_crossfit_f1']:
        ir=next(i for i,c in enumerate(cfg) if c['arm']=='jepa' and c['kind']==kind)
        ie=next(i for i,c in enumerate(cfg) if c['arm']=='blend' and c['kind']==kind)
        diff=td[:,ie,:,0]-td[:,ir,:,0]
        point=np.array(cfg[ie]['per_class'])[:,0]-np.array(cfg[ir]['per_class'])[:,0]
        for j,label in enumerate(CL+['MACRO']):
            vals=diff[:,j] if j<5 else diff.mean(1);value=point[j] if j<5 else point.mean()
            c=ci(vals);pairs.append(dict(rule=kind,target=label,f1_gain=value,ci95_low=c['ci95'][0],ci95_high=c['ci95'][1],valid_draws=c['valid']))
    csvsave('paired_f1_rule_contrasts.csv',pairs)
    op=[]
    for i,c in enumerate(cfg):
        p=np.asarray(c['per_class']);sense=td[:,i,:,1];spec=td[:,i,:,2]
        pden=sense*prev+(1-spec)*(1-prev);nden=spec*(1-prev)+(1-sense)*prev
        ppv=np.divide(sense*prev,pden,out=np.full_like(sense,np.nan),where=pden>0)
        npv=np.divide(spec*(1-prev),nden,out=np.full_like(sense,np.nan),where=nden>0)
        counts=np.array(pts['pooled_confusion_counts'])[i];tp,fp,fn,tn=counts.T
        for j,label in enumerate(CL):
            pointppv=tp[j]/(tp[j]+fp[j]) if tp[j]+fp[j] else None
            pointnpv=tn[j]/(tn[j]+fn[j]) if tn[j]+fn[j] else None
            assert pointppv is None or abs(pointppv-p[j,4])<1e-12
            row=dict(arm=c['arm'],rule=c['kind'],fitting_specificity=c.get('target_specificity'),target=label,
              f1=p[j,0],sensitivity=p[j,1],specificity=p[j,2],ppv=pointppv,npv=pointnpv,
              tp=int(tp[j]),fp=int(fp[j]),fn=int(fn[j]),tn=int(tn[j]))
            for metric,values in [('ppv',ppv[:,j]),('npv',npv[:,j])]:
                cc=ci(values);row.update({metric+'_ci_low':cc['ci95'][0] if cc['ci95'] else None,metric+'_ci_high':cc['ci95'][1] if cc['ci95'] else None,metric+'_valid':cc['valid']})
            op.append(row)
    csvsave('classwise_operating_points.csv',op)
    # Effect sizes, not a claim that a prior shift causally explains calibration.
    cal=list(csv.DictReader((P/'results/a05_calibration/calibration_offsets.csv').open()));effects=[]
    for row in cal:
        sa=float(row['source_offset_point']);ta=float(row['external_offset_point']);delta=float(row['prior_shift_delta_point'])
        residual=ta-sa-delta;prefix='external_minus_source_plus_prior_shift'
        effects.append(dict(arm=row['arm'],target=row['label'],source_offset=sa,target_offset=ta,observed_offset_change=ta-sa,
          prior_log_odds_delta=delta,residual=residual,residual_ci_low=float(row[prefix+'_lo95']),residual_ci_high=float(row[prefix+'_hi95']),
          abs_residual_to_abs_prior_delta=abs(residual/delta) if abs(delta)>=.1 else None))
    csvsave('prior_shift_effect_sizes.csv',effects)
    lookup={c['kind']:c['macro'][0] for c in cfg if c['arm']=='blend' and c['kind']in ['transported_development','retrospective_prevalence_adjusted','external_oracle_f1']}
    recovered=(lookup['retrospective_prevalence_adjusted']-lookup['transported_development'])/(lookup['external_oracle_f1']-lookup['transported_development'])
    # Authenticate original saved probabilities and metadata once.
    study=H/'experiments/sph_component_manuscript_20261026_v1';audit,rows=mapping_and_rows(study)
    base=read(H/'experiments/sph_publication_scope_20261026_v1/external_assessment.json')
    arrays,patients,_=load_predictions(study,base,rows);y=arrays['y'];scores=[arrays['resnet'],arrays['jepa'],(arrays['resnet']+arrays['jepa'])*.5]
    meta=list(csv.DictReader((study/'sources/metadata.csv').open()));byid={r['ECG_ID']:r for r in meta}
    dictionary={int(r['Code']):r['Description'] for r in csv.DictReader((study/'sources/code.csv').open())}
    mapping=read(H/'sph_primary_code_map.json');targets={int(code):label for label,codes in mapping['groups'].items() for code in codes}
    maprows=[dict(source_code=code,source_statement=dictionary[code],research_target=target,clinical_equivalence_adjudicated=False) for code,target in sorted(targets.items())]
    csvsave('code_to_target_map.csv',maprows)
    all_codes={r['ECG_ID']:{int(t.split('+')[0]) for t in r['AHA_Code'].split(';')} for r in meta}
    excluded=[r for r in meta if not any(c in targets for c in all_codes[r['ECG_ID']])];assert len(excluded)==4587
    counter=collections.Counter(c for r in excluded for c in all_codes[r['ECG_ID']])
    csvsave('excluded_statement_breakdown.csv',[dict(source_code=c,source_statement=dictionary[c],records=counter[c],percent_of_excluded=100*counter[c]/4587,
       counts_overlap_across_statements=True) for c in sorted(dictionary) if c<300])
    patterns=collections.Counter(';'.join(map(str,sorted(all_codes[r['ECG_ID']]))) for r in excluded)
    csvsave('excluded_statement_combinations.csv',[dict(primary_code_pattern=k,records=n) for k,n in patterns.most_common()])
    mifs=[];diversity=[]
    for arm,s in zip(['resnet','jepa','blend'],scores):
        mask=(y[:,1]==0)&(s[:,1]>=.5);cc=collections.Counter(c for r,keep in zip(rows,mask) if keep for c in all_codes[r['id']])
        for code in sorted(dictionary):
            if code>=300:continue
            mifs.append(dict(arm=arm,source_code=code,source_statement=dictionary[code],false_positive_records=cc[code],total_annotated_mi_false_positives=int(mask.sum()),
                            percent_of_arm_false_positives=100*cc[code]/mask.sum(),counts_overlap_across_statements=True))
    csvsave('mi_false_positive_statements.csv',mifs)
    for j,label in enumerate(CL):
        pr=scores[0][:,j]>=.5;pj=scores[1][:,j]>=.5;er=pr!=(y[:,j]==1);ej=pj!=(y[:,j]==1)
        diversity.append(dict(target=label,score_correlation=np.corrcoef(scores[0][:,j],scores[1][:,j])[0,1],error_phi=np.corrcoef(er,ej)[0,1],
           decision_disagreement=np.mean(pr!=pj),both_wrong=int((er&ej).sum()),R_only_wrong=int((er&~ej).sum()),J_only_wrong=int((~er&ej).sum()),both_correct=int((~er&~ej).sum())))
    csvsave('retained_pair_diversity.csv',diversity)
    # Per-class new seed outcomes from all83 already authenticated prediction chunks.
    manifest=read(P/'observed_a01_followup_completed_v1/prediction_manifest.json');parts=[]
    for part in manifest['parts']:
        f=P/'observed_a01_followup_completed_v1/prediction_chunks'/part['filename'];assert sha(f)==part['sha256']
        with np.load(f,allow_pickle=False) as z:parts.append({k:z[k].copy() for k in ['ids','patient_id','y','resnet','jepa']})
    ids=np.concatenate([r['ids'] for r in parts]);assert ids.tolist()==[r['id'] for r in rows]
    yy=np.concatenate([r['y'] for r in parts]);assert np.array_equal(yy,y)
    rs=np.concatenate([r['resnet'] for r in parts],axis=1);js=np.concatenate([r['jepa'] for r in parts],axis=1)
    seedclass=[]
    for family,prob in [('ResNet',rs),('JEPA head',js)]:
        for seed,s in zip(SEEDS,prob):
            for j,label in enumerate(CL):
                seedclass.append(dict(family=family,seed=seed,target=label,auroc=roc_auc_score(y[:,j],s[:,j]),ap=average_precision_score(y[:,j],s[:,j]),
                  f1=f1_score(y[:,j],s[:,j]>=.5,zero_division=0),brier=np.mean((s[:,j].astype(float)-y[:,j])**2)))
    csvsave('per_class_seed_results.csv',seedclass)
    inclusion=read(P/'observed_coverage_completed_v2/a02_assessment.json');inc=[]
    for arm in ['resnet','jepa','blend']:
        point=inclusion['points'][arm]['threshold_0_5'];ints=inclusion['conditional_patient_intervals'][arm][0]
        for j,label in enumerate(CL):
            for k,metric in enumerate(inclusion['metrics']):
                cc=ints['per_class'][j][k]
                inc.append(dict(arm=arm,target=label,metric=metric,selected=point['reference']['per_class'][j][k],all_released=point['expanded']['per_class'][j][k],
                  all_minus_selected=point['difference_per_class'][j][k],ci95_low=cc['interval'][0],ci95_high=cc['interval'][1]))
    csvsave('classwise_inclusion_results.csv',inc)
    # Fixed support restriction, retaining the original five-class youngest result.
    age=np.array([int(byid[r['id']]['Age']) for r in rows]);idx=age<40;ys=y[idx];ps=patients[idx]
    supports=ys.sum(0);mask=(supports>=20)&((len(ys)-supports)>=20);assert mask.sum()>=2
    ss=[ScoreStatistics(ys[:,mask],s[idx][:,mask]) for s in scores];points=np.stack([s.evaluate()['macro'] for s in ss])
    unique,inv=np.unique(ps,return_inverse=True);rng=np.random.default_rng(2026102806);draw=np.empty((1000,3,7))
    for b in range(1000):
        w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv]
        draw[b]=np.stack([s.evaluate(w)['macro'] for s in ss])
    restricted={'scope':'additional post hoc fixed-support diagnostic; not the original five-target endpoint',
      'age_stratum':'18-39','minimum_positive_and_negative':20,'classes_retained':[c for c,k in zip(CL,mask) if k],
      'positive_support_all_five':supports,'records':int(idx.sum()),'patients':len(unique),'metrics':['auroc','ap','f1','precision','sensitivity','specificity','brier'],
      'arms':['resnet','jepa','blend'],'points':points,'auroc_ci95':[ci(draw[:,i,0],980) for i in range(3)],
      'blend_minus_jepa_auroc':points[2,0]-points[1,0],'blend_minus_jepa_auroc_ci95':ci(draw[:,2,0]-draw[:,1,0],980),
      'seed':2026102806,'draws':1000,'original_five_class_interval_still_withheld':True}
    save('youngest_support_sensitivity.json',restricted);np.savez_compressed(O/'youngest_support_draws.npz',draws=draw)
    summary={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
      'plan_sha256':sha(V/'analysis_plan.json'),'elapsed_seconds':time.monotonic()-start,
      'post_hoc':True,'original_primary_results_unchanged':True,'no_new_neural_execution':True,
      'component_mean_and_sd':component,'seed202_sensitivity':sensitivities,'model_seed_uncertainty':model,
      'paired_f1_rule_macro':[r for r in pairs if r['target']=='MACRO'],
      'oracle_gap_recovered_by_observed_prior_correction':recovered,
      'prior_shift_blend':[r for r in effects if r['arm']=='blend'],
      'excluded_top_statements':sorted([dict(code=c,description=dictionary[c],records=counter[c]) for c in counter],key=lambda r:-r['records']),
      'youngest_support_sensitivity':restricted,
      'input_hashes':{'a01_assessment':sha(P/'observed_a01_followup_completed_v1/control_assessment.json'),'a03_a04_draws':sha(tf),
       'source_metadata':sha(study/'sources/metadata.csv'),'source_dictionary':sha(study/'sources/code.csv')},
      'outputs':{f.name:sha(f) for f in O.iterdir() if f.is_file()}}
    save('completion.json',summary)
    print('AUTHOR_REVISION_SAVED_SCORE_ANALYSES_COMPLETED',clean({'seconds':summary['elapsed_seconds'],'macro_f1_contrasts':summary['paired_f1_rule_macro'],
      'prior_gap_recovery':recovered,'youngest':restricted['classes_retained'],'youngest_blend_auroc':points[2,0]}),flush=True)
if __name__=='__main__':main()
