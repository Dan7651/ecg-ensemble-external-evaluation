"""Post hoc definition sensitivity from existing scores; no training or clinical claims."""
from pathlib import Path
import csv,gzip,hashlib,json
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score

import argparse
ap=argparse.ArgumentParser(description='Post hoc definition sensitivity from frozen public probabilities; no model or threshold fitting')
ap.add_argument('--release',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('reproduced_definition'))
a=ap.parse_args();R=a.release.resolve();META=a.metadata;P=a.out.resolve()
for folder in ['analysis','supplement']:(P/folder).mkdir(parents=True,exist_ok=True)
CL=['NORM','MI','STTC','CD','HYP'];SEEDS=[101,202,303,404,505]
assert hashlib.sha256(META.read_bytes()).hexdigest()=='c6f8dc5197758a3970aae554f8b6ee884d96656b121879bc0eb1adf367ea6d86'
meta={r['ECG_ID']:r for r in csv.DictReader(META.open())}
mapping=json.loads((R/'data/frozen_primary_code_map.json').read_text())['groups']
def read(name):
 with gzip.open(R/'scores'/name,'rt') as f:return list(csv.DictReader(f))
def labels(ids,remove=()):
 return np.array([[bool(set(int(t.split('+')[0]) for t in meta[k]['AHA_Code'].split(';'))&(set(mapping[c])-set(remove))) for c in CL] for k in ids],np.uint8)
def auc(y,p):return float(roc_auc_score(y,p,average='macro'))
def write(name,rows):
 with (P/'analysis'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
allrows=read('original_first_window_all_25770.csv.gz');rows=[r for r in allrows if r['selected']=='1'];ids=[r['ECG_ID'] for r in rows];y=labels(ids)
def score(rows,arm):return np.array([[r[f'{arm}_{c}'] for c in CL] for r in rows],np.float32)
pr,pj=score(rows,'R42'),score(rows,'J42');pb=(pr+pj)*np.float32(.5)
seedfile=next(f.name for f in (R/'scores').glob('*.gz') if 'seed' in f.name and 'input' not in f.name)
sr=read(seedfile);assert [r['ECG_ID'] for r in sr]==ids
arms=sorted({k.rsplit('_',1)[0] for k in sr[0] if k.rsplit('_',1)[-1] in CL})
print('Seed arms',arms,flush=True)
rs=[score(sr,next(a for a in arms if str(s) in a and a.startswith('R'))) for s in SEEDS]
js=[score(sr,next(a for a in arms if str(s) in a and a.startswith('J'))) for s in SEEDS]
codes=[{int(t.split('+')[0]) for t in meta[k]['AHA_Code'].split(';')} for k in ids]
sole82=np.array([c&set(mapping['CD'])=={82} for c in codes]);sole147=np.array([c&set(mapping['STTC'])=={147} for c in codes])
scenarios=[('Locked map',y,np.ones(len(y),bool)),('Withdraw 82',labels(ids,[82]),np.ones(len(y),bool)),('Withdraw 147',labels(ids,[147]),np.ones(len(y),bool)),('Withdraw both',labels(ids,[82,147]),np.ones(len(y),bool)),('Omit ambiguous',y,~(sole82|sole147))]
results=[];pairs=[]
for name,truth,keep in scenarios:
 yy=truth[keep];ra,ja,ba=[auc(yy,p[keep]) for p in [pr,pj,pb]]
 gg=[]
 for i,s in enumerate(SEEDS):
  j=(i+1)%5;a,b=rs[i][keep],js[j][keep];aa,bb,ab=auc(yy,a),auc(yy,b),auc(yy,(a+b)*np.float32(.5));g=ab-max(aa,bb);gg.append(g)
  pairs.append(dict(scenario=name,resnet_seed=s,jepa_seed=SEEDS[j],resnet_auroc=aa,jepa_auroc=bb,blend_auroc=ab,gain_over_better=g,gain_over_jepa=ab-bb))
 results.append(dict(scenario=name,records=int(keep.sum()),resnet_auroc=ra,jepa_auroc=ja,blend_auroc=ba,blend_minus_jepa=ba-ja,seed_gain_mean=float(np.mean(gg)),seed_gain_sd=float(np.std(gg,ddof=1)),seed_gain_min=float(min(gg)),seed_gain_max=float(max(gg))))
ya=labels([r['ECG_ID'] for r in allrows]);ar,aj=score(allrows,'R42'),score(allrows,'J42');ab=(ar+aj)*np.float32(.5)
results.append(dict(scenario='All 25,770 records',records=len(ya),resnet_auroc=auc(ya,ar),jepa_auroc=auc(ya,aj),blend_auroc=auc(ya,ab),blend_minus_jepa=auc(ya,ab)-auc(ya,aj),seed_gain_mean='',seed_gain_sd='',seed_gain_min='',seed_gain_max=''))
# Reconcile every original mapping stress result against the previously sealed aggregate.
agg=json.loads((R/'data/aggregate_results.json').read_text())
for row,key in zip(results[1:5],['withdraw82','withdraw147','withdraw_both','omit_ambiguous_records']):
 for arm in ['resnet','jepa','blend']:assert abs(row[arm+'_auroc']-agg['mapping_sensitivity'][key]['arms'][arm]['auroc'])<1e-12
write('definition_gain_sensitivity.csv',results);write('five_seed_mapping_gains.csv',pairs)
text='\\begin{center}\\small\n\\begin{tabular}{lrrrr}\\toprule\nDefinition & ECGs & Original blend & $\\Delta$ vs JEPA & Five-pair gain mean [range] \\\\\\midrule\n'
for r in results:
 seed='Not scored' if r['seed_gain_mean']=='' else f"{r['seed_gain_mean']:.5f} [{r['seed_gain_min']:.5f}, {r['seed_gain_max']:.5f}]"
 text+=f"{r['scenario']} & {r['records']:,} & {r['blend_auroc']:.4f} & {r['blend_minus_jepa']:.5f} & {seed} \\\\\n"
text+='\\bottomrule\\end{tabular}\\end{center}\n'
(P/'supplement/definition_gains.tex').write_text(text)
