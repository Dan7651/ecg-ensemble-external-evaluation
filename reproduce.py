"""Rebuild aggregate tables/figures; never read patient data or run a model."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'data/aggregate_results.json').read_text())
OUT=ROOT/'reproduced'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'savefig.dpi':240,'axes.labelcolor':'#233348','text.color':'#233348'})
COLORS=['#196B9E','#DA7338','#21866B']

def csvout(name,header,rows):
    OUT.mkdir(exist_ok=True)
    with (OUT/(name+'.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def texout(name,caption,label,header,rows,note='',wide=False):
    env='table*' if wide else 'table'
    s='\\begin{'+env+'}[!t]\n\\caption{'+caption+'}\\label{'+label+'}\n\\centering\\footnotesize\n'
    s+='\\begin{tabular}{@{}l'+'r'*(len(header)-1)+'@{}}\\toprule\n'
    s+=' & '.join(header)+' \\\\\\midrule\n'
    s+='\n'.join(' & '.join(str(x) for x in row)+' \\\\' for row in rows)
    s+='\n\\bottomrule\\end{tabular}\n'
    if note:s+='\\par\\smallskip\\begin{minipage}{\\columnwidth}\\scriptsize '+note+'\\end{minipage}\n'
    s+='\\end{'+env+'}\n'
    (OUT/(name+'.tex')).write_text(s)

def bare(name,header,rows):
    s='\\begin{center}\\footnotesize\\begin{tabular}{@{}l'+'r'*(len(header)-1)+'@{}}\\toprule\n'
    s+=' & '.join(header)+' \\\\\\midrule\n'
    s+='\n'.join(' & '.join(str(x) for x in row)+' \\\\' for row in rows)
    s+='\n\\bottomrule\\end{tabular}\\end{center}\n'
    (OUT/(name+'.tex')).write_text(s)

def table(name):
    OUT.mkdir(exist_ok=True)
    if name=='original':
        rows=[]
        for arm in DATA['a02']['arms']:
            v=DATA['a02']['points'][arm]['threshold_0_5']['reference']['macro']
            rows.append([{'resnet':'ResNet','jepa':'JEPA head','blend':'Equal blend'}[arm],v[0],v[1],v[2],v[6]])
        csvout(name,['model','macro_auroc','macro_ap','macro_f1_at_0_5','macro_brier'],rows)
        texout(name,'Original Frozen Transfer on the Selected SPH Cohort','tab:original',
          ['Model','AUROC','AP','F1','Brier'],[[r[0]]+[f'{x:.4f}' for x in r[1:]] for r in rows],
          'Original seed 42; F1 at 0.5. The stronger single component is JEPA by AUROC/AP. The originally specified comparator remains ResNet.')
    elif name=='controls':
        d=DATA['a01'];rows=[]
        for family in ['RR','JJ','RJ']:
            a=d['seed_summary'][family]
            rows.append([family,*[a[x][0] for x in ['mean','sample_sd','minimum','maximum']],a['mean'][1],a['mean'][2]])
        csvout(name,['family','gain_auroc_mean','sample_sd','minimum','maximum','gain_ap_mean','gain_f1_mean'],rows)
        texout(name,'Five-Seed Gain Over the Better Component','tab:controls',
          ['Pairs','Mean','SD','Min.','Max.'],[[r[0]]+[f'{x:.5f}' for x in r[1:5]] for r in rows],
          'AUROC gains; five cyclic blocks per family. Blocks share models. JJ varies heads on one shared encoder. No best-seed selection.')
        bare('seed_singles',['Fit','AUROC','AP','F1','Brier'],[[n.replace('_',r'\_')]+[f'{v:.4f}' for v in a] for n,a in zip(d['single_order'],d['singles'])])
        pairs=[]
        for fam in ['RR','JJ','RJ']:
            for i,vals in enumerate(d['seed_summary'][fam]['individual'],1):pairs.append([f'{fam} block {i}']+[f'{v:.5f}' for v in vals])
        bare('pair_gains',['Pair','AUROC gain','AP gain','F1 gain','Brier gain'],pairs)
    elif name=='sensitivity':
        rows=[]
        for arm in DATA['a02']['arms']:
            v=DATA['a02']['points'][arm]['threshold_0_5']
            rows.append([arm,*v['reference']['macro'][:3],*v['expanded']['macro'][:3]])
        csvout(name,['model','selected_auroc','selected_ap','selected_f1','all_auroc','all_ap','all_f1'],rows)
        texout(name,'Cohort Inclusion Sensitivity: Original Models, Unchanged Rules','tab:sensitivity',
          ['Model','Selected area','All area','All AP','All F1'],
          [[{'resnet':'ResNet','jepa':'JEPA','blend':'Blend'}[r[0]],f'{r[1]:.4f}',f'{r[4]:.4f}',f'{r[5]:.4f}',f'{r[6]:.4f}'] for r in rows],
          'Selected: 21,183 recordings; all: 25,770. Added mapped-zero labels are annotation negatives, not adjudicated healthy controls. F1 uses 0.5. Mapping withdrawal is a distinct sensitivity on the selected population.')
    elif name=='thresholds':
        d=DATA['threshold'];rows=[]
        kindnames={'fixed_0_5':'Fixed 0.5','transported_development':'Transported vector',
          'retrospective_prevalence_adjusted':'Prevalence adjustment','patient_crossfit_f1':'Patient cross-fit','external_oracle_f1':'External oracle'}
        # Names are taken from actual frozen assessment, not inferred from ordering.
        configs=[c for c in d['configurations'] if c['arm']=='blend' and c['kind'] in kindnames]
        for c in configs:
            ci=c['conditional_patient_intervals']['macro'][0]['interval']
            rows.append([c['kind'],c['macro'][0],*ci,c['macro'][1],c['macro'][2]])
        csvout(name,['rule','macro_f1','lo95','hi95','macro_sensitivity','macro_specificity'],rows)
        order=['fixed_0_5','transported_development','retrospective_prevalence_adjusted','patient_crossfit_f1','external_oracle_f1']
        rows.sort(key=lambda r:order.index(r[0]))
        texout(name,'Threshold Strategies for the Original Blend','tab:thresholds',
          ['Rule','F1',r'95\% interval'],[[kindnames[r[0]],f'{r[1]:.4f}',f'[{r[2]:.4f}, {r[3]:.4f}]'] for r in rows],
          'The last three strategies use target labels. Cross-fitting is within-target evaluation; oracle performance is optimistic. Intervals refit target thresholds within each patient draw.')
        allrows=[]
        for c in d['configurations']:
            if c['kind'] in kindnames:
                ci=c['conditional_patient_intervals']['macro'][0]['interval']
                allrows.append([c['arm'],kindnames[c['kind']],f"{c['macro'][0]:.4f}",f'[{ci[0]:.4f}, {ci[1]:.4f}]',f"{c['macro'][1]:.4f}",f"{c['macro'][2]:.4f}"])
        bare('threshold_details',['Model','Rule','F1',r'95\% interval','Sens.','Spec.'],allrows)
        sp=[]
        for c in d['configurations']:
            if c['arm']=='blend' and c['kind']=='patient_crossfit_specificity':
                for label,vals in zip(d['class_order'],c['per_class']):
                    sp.append([str(c['target_specificity']),label,f'{vals[1]:.4f}',f'{vals[2]:.4f}'])
        bare('specificity_details',['Fitting target','Class','Sensitivity','Achieved specificity'],sp)
    elif name=='calibration':
        rows=[]
        for r in DATA['calibration_offsets']:
            if r['arm']=='blend':
                pref='external_minus_source_plus_prior_shift'
                rows.append([r['label'],float(r[pref+'_point']),float(r[pref+'_lo95']),float(r[pref+'_hi95'])])
        csvout(name,['class','residual_offset','lo95','hi95'],rows)
        bare('calibration_offsets',['Class','Offset residual',r'Pointwise 95\% interval'],[[r[0],f'{r[1]:.4f}',f'[{r[2]:.4f}, {r[3]:.4f}]'] for r in rows])
    else:raise ValueError(name)

def figure(name):
    if name in ['matched_controls','training_curves','calibration_differences','duration_effects','annotation_sensitivity','confusion_matrices']:
        import additional_figures
        return additional_figures.figure(name)
    OUT.mkdir(exist_ok=True)
    if name=='matched_controls':
        fig,ax=plt.subplots(figsize=(3.5,2.45),layout='constrained')
        for i,fam in enumerate(['RR','JJ','RJ']):
            g=np.array(DATA['a01']['seed_summary'][fam]['individual'])[:,0]
            ax.scatter(np.arange(5)*.06-.12+i,g,s=23,color=COLORS[i],zorder=3)
            ax.plot([i-.25,i+.25],[g.mean()]*2,color=COLORS[i],lw=2.2)
        ax.axhline(0,color='#657485',ls='--',lw=.8)
        ax.set(xticks=[0,1,2],xticklabels=['RR','JJ (shared encoder)','RJ'],ylabel='AUROC gain over better component')
        ax.set_ylim(-.0011,.0072);ax.grid(axis='y',alpha=.18)
    elif name=='transfer_diagnostics':
        fig,axs=plt.subplots(1,2,figsize=(7.1,2.6),layout='constrained')
        ax=axs[0]
        for i,arm in enumerate(DATA['a02']['arms']):
            p=DATA['a02']['points'][arm]['threshold_0_5']
            ax.plot([0,1],[p['reference']['macro'][0],p['expanded']['macro'][0]],'-o',color=COLORS[i],label=['ResNet','JEPA head','Blend'][i],ms=4)
        ax.set(xticks=[0,1],xticklabels=['Selected\n21,183 ECGs','All released\n25,770 ECGs'],ylabel='Macro AUROC',title='A  Cohort inclusion')
        ax.legend(frameon=False,fontsize=7);ax.set_ylim(.903,.951);ax.grid(axis='y',alpha=.18)
        ax=axs[1]
        names=['fixed_0_5','transported_development','retrospective_prevalence_adjusted','patient_crossfit_f1','external_oracle_f1']
        labels=['Fixed 0.5','Transported','Prior-adjusted*','Patient cross-fit*','Oracle*']
        configs={c['kind']:c for c in DATA['threshold']['configurations'] if c['arm']=='blend'}
        for i,k in enumerate(names):
            c=configs[k];v=c['macro'][0];lo,hi=c['conditional_patient_intervals']['macro'][0]['interval']
            ax.errorbar(v,4-i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[2] if i<2 else '#8964A8',capsize=2,ms=4)
        ax.set(yticks=range(5),yticklabels=labels[::-1],xlabel='Macro F1',title='B  Threshold strategy')
        ax.set_xlim(.53,.72);ax.grid(axis='x',alpha=.18)
        ax.text(0,-.26,'* Uses target labels; retrospective diagnostic.',transform=ax.transAxes,fontsize=7)
    elif name=='calibration_review':
        fig,axs=plt.subplots(2,3,figsize=(7.16,3.9),layout='constrained');axs=axs.ravel()
        supports=DATA['a02']['denominators']['selected_records']['positive_counts']
        for ax,label,npos in zip(axs,DATA['classes'],supports):
            ax.plot([0,1],[0,1],ls='--',lw=.7,color='#8895A2')
            for row in DATA['calibration_bins']:
                if row['class']!=label:continue
                x,y=row['mean_probability'],row['fraction_positive']
                ax.plot(x,y,'o',color='#067D86',ms=2+np.log10(row['n']),zorder=3)
                if row['ci95'] is not None:ax.vlines(x,*row['ci95'],color='#067D86',lw=.7,zorder=2)
            ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Mean probability',ylabel='Positive fraction')
            ax.set_title(f'{label} (positive n = {npos:,})',fontsize=8)
        axs[5].axis('off');axs[5].text(0,.95,'Unchanged equal-probability blend\n10 equal-width probability bins\nMarker size increases with bin support\nMI/HYP counts: Supplement Section S3\nPointwise 95% patient intervals\n1,000 patient-cluster resamples\nNo probability recalibration',fontsize=7,va='top',linespacing=1.7)
    else:raise ValueError(name)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');p.add_argument('--table',choices=['original','controls','sensitivity','thresholds','calibration']);p.add_argument('--figure',choices=['matched_controls','transfer_diagnostics','calibration_review','training_curves','calibration_differences','duration_effects','annotation_sensitivity','confusion_matrices']);a=p.parse_args()
    for name in (['original','controls','sensitivity','thresholds','calibration'] if a.all else [a.table] if a.table else []):table(name)
    for name in (['matched_controls','transfer_diagnostics','calibration_review','training_curves','calibration_differences','duration_effects','annotation_sensitivity','confusion_matrices'] if a.all else [a.figure] if a.figure else []):figure(name)
    print('AGGREGATE_REPRODUCTION_COMPLETE',OUT)
if __name__=='__main__':main()
