from pathlib import Path
import gzip,csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
import argparse
ap=argparse.ArgumentParser(description='Empirical ROC/PR from unchanged public scores and pinned source metadata')
ap.add_argument('--release',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('reproduced_curves_v12'))
a=ap.parse_args();R=a.release.resolve();P=a.out.resolve()
for folder in ['analysis','figures']:(P/folder).mkdir(parents=True,exist_ok=True)
assert hashlib.sha256(a.metadata.read_bytes()).hexdigest()=='c6f8dc5197758a3970aae554f8b6ee884d96656b121879bc0eb1adf367ea6d86'
meta={r['ECG_ID']:r for r in csv.DictReader(a.metadata.open())}
with gzip.open(R/'scores/original_first_window_all_25770.csv.gz','rt') as f:rows=[r for r in csv.DictReader(f) if r['selected']=='1']
cl=['NORM','MI','STTC','CD','HYP'];m=json.loads((R/'data/frozen_primary_code_map.json').read_text())['groups']
y=np.array([[bool({int(t.split('+')[0]) for t in meta[r['ECG_ID']]['AHA_Code'].split(';')}&set(m[c])) for c in cl] for r in rows],np.uint8)
scores=[np.array([[r[f'{a}_{c}'] for c in cl] for r in rows],np.float32) for a in ['R42','J42']];scores.append((scores[0]+scores[1])*np.float32(.5))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'pdf.fonttype':42,'svg.fonttype':'none'})
coords=[]
for kind in ['roc','pr']:
 fig,ax=plt.subplots(1,5,figsize=(7.5,1.75));fig.subplots_adjust(left=.055,right=.99,bottom=.26,top=.83,wspace=.38)
 for j,c in enumerate(cl):
  for p,col,label in zip(scores,['#196B9E','#DA7338','#21866B'],['ResNet','JEPA head','Equal blend']):
   if kind=='roc':x,z,_=roc_curve(y[:,j],p[:,j],drop_intermediate=False)
   else:z,x,_=precision_recall_curve(y[:,j],p[:,j])
   ax[j].plot(x,z,lw=.9,color=col,label=label);coords.extend((kind,c,label,float(a),float(b)) for a,b in zip(x,z))
  ax[j].set(title=c,xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1]);ax[j].grid(alpha=.12)
  ax[j].set_xlabel('False-positive rate' if kind=='roc' else 'Recall')
  if j==0:ax[j].set_ylabel('Sensitivity' if kind=='roc' else 'Precision')
  if kind=='roc':ax[j].plot([0,1],[0,1],'--',lw=.5,color='#bbbbbb')
 fig.legend(*ax[-1].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=6.5,bbox_to_anchor=(.5,1.03))
 for ext in ['pdf','svg','png']:fig.savefig(P/'figures'/f'{kind}_curves.{ext}',dpi=300)
 plt.close(fig)
with (P/'analysis/empirical_curve_coordinates.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['curve','target','model','x','y']);w.writerows(coords)
print('RESTORED_EMPIRICAL_ROC_PR',len(rows),len(coords))
