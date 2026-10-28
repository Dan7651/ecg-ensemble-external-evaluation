"""Independent arithmetic checks of archived artifacts; no model/checkpoint loading."""
from pathlib import Path
import csv,hashlib,json,itertools
import numpy as np
H=Path(__file__).resolve().parent;V=H/'experiments/sph_author_revision_20261028_v8';P=H/'experiments/sph_prospective_extension_20261027_v1';O=V/'analysis'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=lambda n:list(csv.DictReader((O/n).open()))
c=read(O/'completion.json')
for f,s in c['outputs'].items():assert sha(O/f)==s,f
a=read(P/'observed_a01_followup_completed_v1/control_assessment.json');m=np.array(a['singles']);g=np.array(a['primary_gains']);e=np.array(a['primary_ensembles']).reshape(3,5,4)
for r in rows('component_seed_summary.csv'):
 j=a['metrics'].index(r['metric']);z=m[:5,j] if r['family']=='ResNet' else m[5:,j]
 assert abs(float(r['mean'])-z.mean())<1e-13 and abs(float(r['sample_sd'])-z.std(ddof=1))<1e-13
with np.load(P/'results/a03_a04_thresholds/patient_draws.npz',allow_pickle=False) as z:d=z['per_class_metrics'];p=z['external_prevalence']
t=read(P/'results/a03_a04_thresholds/assessment.json');cfg=t['configurations'];pts=read(P/'results/a03_a04_thresholds/point_estimates.json')
for r in rows('paired_f1_rule_contrasts.csv'):
 i=next(i for i,x in enumerate(cfg) if x['arm']=='blend' and x['kind']==r['rule']);j=next(i for i,x in enumerate(cfg) if x['arm']=='jepa' and x['kind']==r['rule']);delta=d[:,i,:,0]-d[:,j,:,0]
 vals=delta.mean(1) if r['target']=='MACRO' else delta[:,['NORM','MI','STTC','CD','HYP'].index(r['target'])]
 lo,hi=np.percentile(vals,[2.5,97.5]);assert abs(lo-float(r['ci95_low']))<1e-12 and abs(hi-float(r['ci95_high']))<1e-12
for r in rows('classwise_operating_points.csv'):
 tp,fp,fn,tn=[int(r[k]) for k in ['tp','fp','fn','tn']];assert tp+fp+fn+tn==21183
 for key,v in [('f1',2*tp/(2*tp+fp+fn)),('ppv',tp/(tp+fp) if tp+fp else None),('npv',tn/(tn+fn) if tn+fn else None)]:
  assert v is None or abs(float(r[key])-v)<1e-12
 i=next(i for i,x in enumerate(cfg) if x['arm']==r['arm'] and x['kind']==r['rule'] and str(x.get('target_specificity') or '')==r['fitting_specificity']);j=['NORM','MI','STTC','CD','HYP'].index(r['target'])
 se,sp,pr=d[:,i,j,1],d[:,i,j,2],p[:,j];npv=sp*(1-pr)/(sp*(1-pr)+(1-se)*pr);ci=np.percentile(npv,[2.5,97.5]);assert np.allclose(ci,[float(r['npv_ci_low']),float(r['npv_ci_high'])],atol=1e-12)
for r in rows('retained_pair_diversity.csv'):
 n=sum(int(r[k]) for k in ['both_wrong','R_only_wrong','J_only_wrong','both_correct']);assert n==21183
 assert abs(float(r['decision_disagreement'])-(int(r['R_only_wrong'])+int(r['J_only_wrong']))/n)<1e-12
assert sum(int(r['records']) for r in rows('excluded_statement_combinations.csv'))==4587
assert len(rows('code_to_target_map.csv'))==25 and len(rows('per_class_seed_results.csv'))==50
mod=read(O/'model_seed_uncertainty.json')
with np.load(O/'model_resampling_draws.npz') as z:
 assert np.allclose(np.percentile(z['model_seed_bootstrap'],[2.5,97.5],axis=0),mod['model_seed_bootstrap_ci95'])
 assert np.allclose(np.mean(z['model_label_permutations']>=np.array(mod['permutation_statistic'])-1e-15,axis=0),mod['permutation_upper_tail_fraction'])
y=read(O/'youngest_support_sensitivity.json')
with np.load(O/'youngest_support_draws.npz') as z:
 assert z['draws'].shape==(1000,3,7);ci=np.percentile(z['draws'][:,2,0]-z['draws'][:,1,0],[2.5,97.5]);assert np.allclose(ci,y['blend_minus_jepa_auroc_ci95']['ci95'])
r=dict(status='passed',source_sha256=sha(__file__),analysis_completion_sha256=sha(O/'completion.json'),checks=['all_output_hashes','component_mean_sample_sd','three_paired_f1_intervals','145_operating_point_counts_and_npv_intervals','error_disagreement_counts','excluded_patterns_total','25_code_map','50_seed_class_results','model_resampling_percentiles','youngest_support_percentiles'],scope='Independent saved-artifact arithmetic; no real model execution or full underlying bootstrap replay')
(V/'independent_verification.json').write_text(json.dumps(r,indent=2));print('AUTHOR_REVISION_INDEPENDENT_ARITHMETIC_PASSED')
