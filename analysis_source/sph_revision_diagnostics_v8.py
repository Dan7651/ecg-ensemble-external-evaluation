"""Additional author-interpretation figure from verified saved-result tables only."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
V=Path(__file__).resolve().parent/'experiments/sph_author_revision_20261028_v8';A=V/'analysis';F=V/'figures'
rows=lambda n:list(csv.DictReader((A/n).open()));colors=['#196B9E','#DA7338','#21866B'];CL=['NORM','MI','STTC','CD','HYP']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(9,5.5),layout='constrained')
s=rows('seed202_sensitivity.csv');ax=axs[0,0]
for i,metric in enumerate(['f1_at_0_5','brier']):
 rs=[next(r for r in s if r['family']=='RJ' and r['metric']==metric and r['sensitivity']==scope) for scope in ['all_five','omit_ResNet202_only','omit_both_seed202']]
 x=np.arange(3)+(i-.5)*.18;y=np.array([float(r['gain_mean']) for r in rs]);lo=[float(r['patient_ci_low']) for r in rs];hi=[float(r['patient_ci_high']) for r in rs];ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=2,color=colors[i],label=['F1 at 0.5','Brier (lower is better)'][i],ms=4)
ax.axhline(0,color='grey',ls='--',lw=.7);ax.set(title='A  Seed influence: mixed-family gains',xticks=range(3),xticklabels=['All five','Omit R202','Omit both 202'],ylabel='Gain over better component');ax.legend(frameon=False,fontsize=7)
ax=axs[0,1];rs=[r for r in rows('prior_shift_effect_sizes.csv') if r['arm']=='blend'];x=np.arange(5)
ax.bar(x-.17,[float(r['prior_log_odds_delta']) for r in rs],.32,color=colors[0],label='Prior-log-odds prediction');ax.bar(x+.17,[float(r['observed_offset_change']) for r in rs],.32,color=colors[2],label='Observed offset change');ax.axhline(0,color='grey',lw=.6);ax.set(title='B  Calibration effect-size comparison',xticks=x,xticklabels=CL,ylabel='Logit offset change');ax.legend(frameon=False,fontsize=7)
ax=axs[1,0];rs=[r for r in rows('paired_f1_rule_contrasts.csv') if r['target']=='MACRO'];y=np.array([float(r['f1_gain']) for r in rs]);lo=np.array([float(r['ci95_low']) for r in rs]);hi=np.array([float(r['ci95_high']) for r in rs]);ax.errorbar(y,range(3),xerr=[y-lo,hi-y],fmt='o',capsize=3,color=colors[2]);ax.axvline(0,color='grey',ls='--',lw=.7);ax.set(title='C  Original pair: threshold-dependent F1',yticks=range(3),yticklabels=['Fixed 0.5','Target oracle','Patient cross-fit'],xlabel='Blend minus JEPA F1, paired 95% CI');ax.invert_yaxis()
ax=axs[1,1];rs=[r for r in rows('classwise_operating_points.csv') if r['arm']=='blend' and r['rule']=='patient_crossfit_specificity' and r['target']=='MI'];x=np.arange(3)
for i,metric in enumerate(['sensitivity','ppv','npv']):ax.plot(x,[float(r[metric]) for r in rs],'o-',lw=1,ms=4,color=colors[i],label=metric.upper() if metric!='sensitivity' else 'Sensitivity')
ax.set(title='D  MI predictive values at 1.23% prevalence',xticks=x,xticklabels=['0.90','0.95','0.99'],xlabel='Fitting specificity target',ylabel='Observed held-out metric',ylim=(0,1.05));ax.legend(frameon=False,fontsize=7)
for ax in axs.flat:ax.grid(axis='y',alpha=.15)
for ext in ['pdf','svg','png']:fig.savefig(F/('revision_diagnostics.'+ext),bbox_inches='tight',**({'dpi':220} if ext=='png' else {}))
print('VERIFIED_REVISION_DIAGNOSTICS_EXPORTED')
