"""Experiment figures from aggregate tables and archived fit histories only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'data/aggregate_results.json').read_text())
OUT=ROOT/'reproduced'
C=['#196B9E','#DA7338','#21866B']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.6,'axes.spines.top':False,'axes.spines.right':False,
 'svg.fonttype':'none','pdf.fonttype':42,'axes.labelcolor':'#233348','text.color':'#233348'})

def error(ax,x,y,lo,hi,**kw):
    ax.hlines(y,lo,hi,**kw);ax.plot(x,y,'o',color=kw.get('color'),ms=3.3,zorder=3)

def figure(name):
    OUT.mkdir(exist_ok=True)
    if name=='matched_controls':
        fig,axs=plt.subplots(1,3,figsize=(7.16,2.15),layout='constrained')
        for j,(ax,title) in enumerate(zip(axs,['A  AUROC gain','B  AP gain','C  F1 gain at 0.5'])):
            for i,fam in enumerate(['RR','JJ','RJ']):
                allg=np.array([r['gain'][j] for r in D['a01']['all_45_sensitivity_pairs'] if r['family']==fam])
                ax.scatter(i+np.linspace(-.22,.22,len(allg)),allg,s=10,facecolors='none',edgecolors=C[i],alpha=.4,lw=.6)
                g=np.array(D['a01']['seed_summary'][fam]['individual'])[:,j]
                ax.scatter(i+np.linspace(-.12,.12,5),g,s=18,color=C[i],zorder=3)
                ax.plot([i-.28,i+.28],[g.mean()]*2,color=C[i],lw=1.6)
            ax.axhline(0,color='#7b8794',ls='--',lw=.7);ax.grid(axis='y',alpha=.15)
            ax.set(xticks=[0,1,2],xticklabels=['RR','JJ','RJ'],title=title)
        axs[0].set_ylabel('Blend minus better component')
    elif name=='training_curves':
        from matplotlib.lines import Line2D
        fig,axs=plt.subplots(1,2,figsize=(7.16,1.85),layout='constrained')
        styles=['-','--','-.',':',(0,(3,1,1,1))]
        for col,fam in enumerate(['resnet','jepa_head']):
            color=C[0] if col==0 else C[2]
            for style,r in zip(styles,[x for x in D['a01_training_histories'] if x['family']==fam]):
                h=r['epochs'];x=[q['epoch'] for q in h];loss=[q['train_bce'] for q in h];auc=[q['metrics']['macro_auroc'] for q in h]
                axs[0].plot(x,loss,color=color,lw=.8,ls=style,alpha=.85)
                axs[1].plot(x,auc,color=color,lw=.8,ls=style,alpha=.85)
                best=x.index(r['selected_epoch']);axs[1].plot(x[best],auc[best],'o',mfc='white',mec=color,ms=2.7)
        axs[0].set(ylabel='Training BCE',title='A  Fitting trajectories');axs[1].set(ylabel='Development AUROC',title='B  Checkpoint selection')
        for ax in axs:ax.set_xlabel('Completed epoch');ax.grid(alpha=.13)
        axs[0].legend(handles=[Line2D([],[],color=C[0],label='ResNet'),Line2D([],[],color=C[2],label='JEPA head')],frameon=False,fontsize=6.3)
        axs[1].legend(handles=[Line2D([],[],color='#666',ls=st,label=str(seed)) for st,seed in zip(styles,[101,202,303,404,505])],ncol=3,frameon=False,fontsize=5.8,loc='lower right')
    elif name=='calibration_differences':
        fig,axs=plt.subplots(1,2,figsize=(7.16,2.35),layout='constrained');labels=['Macro']+D['classes']
        for i,(key,col,lab) in enumerate([('blend_minus_resnet',C[0],'Blend − ResNet'),('blend_minus_jepa',C[2],'Blend − JEPA')]):
            r=D['brier']['contrasts'][key];x=[r['macro_point']]+r['per_class_point'];ci=np.column_stack([r['macro_ci95'],np.array(r['per_class_ci95'])])
            for k in range(6):error(axs[0],x[k],k+(i-.5)*.18,ci[0,k],ci[1,k],color=col,lw=.8)
            axs[0].plot([],[],'o-',color=col,label=lab,ms=3)
        axs[0].set(yticks=range(6),yticklabels=labels,xlabel='Brier difference (negative favors blend)',title='A  Paired probability error');axs[0].invert_yaxis();axs[0].legend(frameon=False,fontsize=6.5,loc='lower left')
        pref='external_minus_source_plus_prior_shift'
        rows=[r for r in D['calibration_offsets'] if r['arm']=='blend']
        for i,r in enumerate(rows):error(axs[1],float(r[pref+'_point']),i,float(r[pref+'_lo95']),float(r[pref+'_hi95']),color='#8964A8',lw=1)
        axs[1].set(yticks=range(5),yticklabels=D['classes'],xlabel='Offset residual beyond source + prior shift',title='B  Calibration departure');axs[1].invert_yaxis()
        for ax in axs:ax.axvline(0,color='#7b8794',ls='--',lw=.7);ax.grid(axis='x',alpha=.15)
    elif name=='duration_effects':
        fig,axs=plt.subplots(1,4,figsize=(7.16,1.6),layout='constrained')
        for ax,j,title in zip(axs,[0,1,2,6],['AUROC','AP','F1 at 0.5','Brier']):
            for i,(key,col) in enumerate([('all_selected',C[0]),('long_only',C[2])]):
                r=D['a06']['points']['blend'][key];ci=D['a06']['conditional_patient_intervals']['blend'][i]['macro'][j]['interval']
                error(ax,r['difference_macro'][j],i,*ci,color=col,lw=1.1)
            ax.axvline(0,color='#7b8794',ls='--',lw=.7);ax.set(title=title,yticks=[0,1],yticklabels=['All selected','Long only'] if j==0 else ['', '']);ax.invert_yaxis();ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2));ax.grid(axis='x',alpha=.15)
    elif name=='annotation_sensitivity':
        fig,axs=plt.subplots(1,2,figsize=(7.16,2.2),layout='constrained')
        variants=['Locked','Withdraw 82','Withdraw 147','Withdraw both','Omit ambiguous']
        keys=[None,'withdraw82','withdraw147','withdraw_both','omit_ambiguous_records']
        actual=list(D['mapping_sensitivity']);keys[3]=next(k for k in actual if 'both' in k);keys[4]=next(k for k in actual if 'omit' in k)
        for i,arm in enumerate(['resnet','jepa','blend']):
            vals=[D['a02']['points'][arm]['threshold_0_5']['reference']['macro'][0]]+[D['mapping_sensitivity'][k]['arms'][arm]['auroc'] for k in keys[1:]]
            axs[0].plot(range(5),vals,'-o',color=C[i],ms=3,lw=.9,label=['ResNet','JEPA','Blend'][i])
        axs[0].set(xticks=range(5),xticklabels=['Locked','−82','−147','−82/147','Omit'],ylabel='Macro AUROC',title='A  Annotation stress tests');axs[0].legend(frameon=False,fontsize=6.5);axs[0].grid(axis='y',alpha=.15)
        nc=D['a02']['points']['blend']['norm_confusion']
        counts=[(nc[k]['0.5']['fp'],nc[k]['0.5']['negative_records']) for k in ['selected_only','all_records','added_only']];rates=[a/b for a,b in counts]
        axs[1].bar(range(3),rates,color=[C[0],C[2],'#8964A8'],width=.55)
        for i,((a,b),r) in enumerate(zip(counts,rates)):axs[1].text(i,r+.025,f'{a:,}/{b:,}\n{r:.3f}',ha='center',fontsize=6.8)
        axs[1].set(xticks=range(3),xticklabels=['Selected','All released','Added only'],ylabel='Mapped-label NORM FPR at 0.5',ylim=(0,.88),title='B  Inclusion effect');axs[1].grid(axis='y',alpha=.15)
    elif name=='confusion_matrices':
        fig,axs=plt.subplots(1,5,figsize=(10,2.25),layout='constrained')
        for ax,(label,counts) in zip(axs,D['original_blend_confusion_counts'].items()):
            a=np.asarray(counts);rates=a/a.sum(1,keepdims=True)
            for i in range(2):
                for j in range(2):ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=plt.cm.Blues(rates[i,j]),edgecolor='white'))
            ax.set_xlim(-.5,1.5);ax.set_ylim(1.5,-.5);ax.set_aspect('equal')
            for i in range(2):
                for j in range(2):ax.text(j,i,f'{a[i,j]:,}\n({rates[i,j]:.3f})',ha='center',va='center',fontsize=7,color='white' if rates[i,j]>.5 else '#233348')
            ax.set(title=label,xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive']);ax.set_xlabel('Predicted at 0.5')
        axs[0].set_ylabel('Mapped reference label')
    else:raise ValueError(name)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',**({'dpi':240} if ext=='png' else {}))
    plt.close(fig)
