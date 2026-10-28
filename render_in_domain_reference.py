from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent/'reproduced_followup';(P/'figures').mkdir(exist_ok=True,parents=True);a=json.loads((P.parent/'followup_tables/in_domain_assessment.json').read_text())['partitions']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
f,ax=plt.subplots(1,3,figsize=(10.5,3.2));col=['#0072B2','#CC79A7','#009E73']
x=np.arange(2)
for i,arm in enumerate(['ResNet','JEPA','Blend']):ax[0].bar(x+(i-1)*.23,[a[k]['original_metrics'][i][0] for k in ['9','10']],width=.22,color=col[i],label=arm)
ax[0].set(xticks=x,xticklabels=['Fold 9','Fold 10'],ylabel='Macro AUROC',ylim=(.9,.94),title='Fixed original pair');ax[0].legend(frameon=False,fontsize=8)
for i,fam in enumerate(['RR','JJ','RJ']):
 for j,k in enumerate(['9','10']):
  values=np.array(a[k]['seed_gains'][fam]['all'])[:,0];ax[1].scatter(np.full(5,j)+(i-1)*.2,values,s=18,color=col[i],alpha=.75,label=fam if j==0 else None);ax[1].plot(j+(i-1)*.2,values.mean(),'_',color='black',ms=10)
ax[1].axhline(0,color='gray',lw=.5);ax[1].set(xticks=x,xticklabels=['Fold 9','Fold 10'],ylabel='AUROC gain over better component',title='Five fixed pair blocks');ax[1].legend(frameon=False,fontsize=8)
for k,marker in [('9','o-'),('10','s--')]:ax[2].plot(range(5),[r['slope'] for r in a[k]['calibration']['blend42']],marker,label='Fold '+k)
ax[2].axhline(1,color='gray',lw=.6);ax[2].set(xticks=range(5),xticklabels=['NORM','MI','STTC','CD','HYP'],ylabel='Joint logistic calibration slope',title='Original blend calibration');ax[2].legend(frameon=False,fontsize=8)
f.tight_layout()
for ext in ['svg','pdf','png']:f.savefig(P/'figures'/('in_domain_reference.'+ext),dpi=180)
