"""Reproduce the input-view table/figure from public-record-ID score exports.

python reproduce_input_view.py --metadata /path/to/SPH/metadata.csv
Add --bootstrap 2000 for the full conditional patient intervals (CPU, no model).
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from saved_score_statistics import ScoreStatistics

ROOT=Path(__file__).resolve().parent
CL=['NORM','MI','STTC','CD','HYP'];SEEDS=[101,202,303,404,505]
GROUPS=['original_RR','JJ','original_RJ','view_RR','view_RJ']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'reproduced_input_view');ap.add_argument('--bootstrap',type=int,choices=[0,2000],default=0)
    args=ap.parse_args();args.out.mkdir(exist_ok=True)
    assert hashlib.sha256(args.metadata.read_bytes()).hexdigest()=='c6f8dc5197758a3970aae554f8b6ee884d96656b121879bc0eb1adf367ea6d86'
    mapping=json.loads((ROOT/'data/frozen_primary_code_map.json').read_text())['groups']
    metadata={r['ECG_ID']:r for r in csv.DictReader(args.metadata.open())}
    def load(name):
        with gzip.open(ROOT/'scores'/name,'rt') as f:return list(csv.DictReader(f))
    old=load('matched_seed_selected_21183.csv.gz');new=load('input_view_five_seed_selected_21183.csv.gz')
    assert [r['ECG_ID'] for r in old]==[r['ECG_ID'] for r in new]
    ids=[r['ECG_ID'] for r in new];assert len(set(ids))==21183
    y=np.array([[bool(set(int(t.split('+')[0]) for t in metadata[key]['AHA_Code'].split(';'))&set(mapping[c])) for c in CL] for key in ids],np.uint8)
    def extract(rows,prefix):return [np.array([[r[f'{prefix}{seed}_{c}'] for c in CL] for r in rows],np.float32) for seed in SEEDS]
    scores=extract(old,'R')+extract(new,'V')+extract(old,'J');definitions=[]
    for group in GROUPS:
        for i in range(5):
            j=(i+1)%5;a=i if group.startswith('original') else 10+i if group=='JJ' else 5+i
            b=j if group=='original_RR' else 5+j if group=='view_RR' else 10+j
            definitions.append((group,i,j,a,b))
    scores += [(scores[a]+scores[b])*.5 for _,_,_,a,b in definitions]
    stats=[ScoreStatistics(y,p) for p in scores]
    point=np.array([s.evaluate()['macro'][[0,1,2,6]] for s in stats])
    for p,row in zip(scores,point):
        independent=[roc_auc_score(y,p,average='macro'),average_precision_score(y,p,average='macro'),f1_score(y,p>=.5,average='macro',zero_division=0),((p.astype(float)-y)**2).mean()]
        np.testing.assert_allclose(row,independent,atol=1e-12,rtol=0)
    def gains(point):
        rows=[]
        for k,(_,_,_,a,b) in enumerate(definitions):
            g=point[15+k]-np.maximum(point[a],point[b]);g[3]=min(point[a,3],point[b,3])-point[15+k,3];rows.append(g)
        return np.array(rows).reshape(5,5,4)
    def contrasts(point):
        g=gains(point).mean(1)
        return np.stack([g[4]-g[2],g[4]-g[3],g[4]-g[1],(point[5:10]-point[:5]).mean(0)])
    gain=gains(point)
    with (args.out/'pairs.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['family','seed_a','seed_b','auroc','ap','f1','brier','gain_auroc','gain_ap','gain_f1','gain_brier'])
        for k,(family,i,j,_,_) in enumerate(definitions):w.writerow([family,SEEDS[i],SEEDS[j],*point[15+k],*gain.reshape(25,4)[k]])
    plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.7),layout='constrained')
    order=[0,2,3,4,1];labels=['Original R+R','Original R+J','Input-matched R+R','Input-matched R+J','J+J'];colors=['#7b8794','#8f70b7','#4296a4','#087f73','#bb843d']
    for c,ax in enumerate(axs.flat):
        for k,(index,color) in enumerate(zip(order,colors)):
            ax.scatter(gain[index,:,c],k+np.linspace(-.14,.14,5),color=color,s=18,alpha=.8,zorder=3)
            ax.plot([gain[index,:,c].mean()]*2,[k-.25,k+.25],color=color,lw=2.3,zorder=4)
        ax.axvline(0,color='#555555',ls='--',lw=.8);ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
        ax.set_xlabel(['AUROC gain','AP gain','F1@0.5 gain','Brier gain (lower error is better)'][c])
        ax.set_title(['(a) Discrimination','(b) Precision-recall','(c) Fixed-threshold decisions','(d) Probability error'][c],loc='left',fontweight='bold',fontsize=9);ax.tick_params(labelsize=7)
    for ext in ['pdf','svg','png']:fig.savefig(args.out/('input_view_control.'+ext),dpi=220)
    plt.close(fig)
    result=dict(singles=point[:15].reshape(3,5,4).tolist(),pair_metrics=point[15:].reshape(5,5,4).tolist(),gains=gain.tolist(),contrasts=contrasts(point).tolist())
    if args.bootstrap:
        patients=[metadata[key]['Patient_ID'] for key in ids];unique,inv=np.unique(patients,return_inverse=True)
        assert len(unique)==20401
        rng=np.random.default_rng(2026102801);draws=[]
        for b in range(2000):
            w=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv]
            draws.append(np.array([s.evaluate(w)['macro'][[0,1,2,6]] for s in stats]))
            if (b+1)%100==0:print('PATIENT_DRAWS',b+1,2000,flush=True)
        draws=np.asarray(draws);contrast=np.asarray([contrasts(d) for d in draws]);np.savez_compressed(args.out/'patient_bootstrap_draws.npz',macro=draws,contrasts=contrast)
        result['ci95']=np.percentile(contrast,[2.5,97.5],axis=0).tolist()
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('INPUT_VIEW_SCORES_TABLES_AND_FIGURE_REPRODUCED',args.out)

if __name__=='__main__':main()
