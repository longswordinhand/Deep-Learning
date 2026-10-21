from pathlib import Path
import json, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'journal_extension'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
TAB=ROOT/'journal_extension'/'tables'; TAB.mkdir(parents=True,exist_ok=True)
phase1=json.load(open(ROOT/'journal_extension/results/phase1_active_label_budget.json'))
phase2=json.load(open(ROOT/'journal_extension/results/phase2_plantwild_active_label_budget.json'))
pdconf=json.load(open(ROOT/'journal_extension/results/phase4_conformal_recovery_plantdoc.json'))
pwconf=json.load(open(ROOT/'journal_extension/results/phase4_conformal_recovery_plantwild.json'))
apple=json.load(open(ROOT/'journal_extension/results/phase6_apple_third_domain.json'))
p11=json.load(open(ROOT/'journal_extension/results/phase11_publication_pooled_metrics.json'))

# Figure 1
tasks=['Corn->PlantDoc','Corn->PlantWild','Apple->PlantDoc']
source_acc=np.array([0.9628,0.9628,apple['source_test']['accuracy']])
field_acc=np.array([phase1['baseline']['accuracy'],phase2['baseline']['accuracy'],apple['field_baseline']['accuracy']])
source_cov=np.array([pdconf['aggregate']['0.1']['0']['coverage_mean'],pwconf['aggregate']['0.1']['0']['coverage_mean'],apple['conformal']['aggregate']['0']['coverage_mean']])
with open(TAB/'fig1_transfer_failure_source.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['task','source_accuracy','field_accuracy','source_only_field_coverage_nominal_0.90'])
    w.writerows(zip(tasks,source_acc,field_acc,source_cov))
fig=plt.figure(figsize=(7.2,6.4),constrained_layout=True); gs=fig.add_gridspec(2,2,height_ratios=[0.85,1.0])
ax=fig.add_subplot(gs[0,:]); ax.axis('off')
boxes=[
    (0.02,0.14,'Curated source\nPlantVillage'),
    (0.28,0.22,'Frozen DINOv2-small\n+ source-selected probe'),
    (0.59,0.18,'Real-field target\nPlantDoc / PlantWild'),
    (0.835,0.15,'Field labels\nAdapt | Calibrate'),
]
for x,w,text in boxes:
    p=FancyBboxPatch((x,0.33),w,0.38,boxstyle='round,pad=0.015',transform=ax.transAxes,fill=False,lw=1.4); ax.add_patch(p)
    ax.text(x+w/2,0.52,text,ha='center',va='center',fontsize=8.2,transform=ax.transAxes)
# Keep arrows entirely in the whitespace between boxes; do not let arrowheads enter boxes.
arrow_gap=0.012
for (x,w,_),(nx,_,__) in zip(boxes[:-1],boxes[1:]):
    ax.add_patch(FancyArrowPatch((x+w+arrow_gap,0.52),(nx-arrow_gap,0.52),arrowstyle='-|>',mutation_scale=12,lw=1.2,transform=ax.transAxes))
ax.text(0.5,0.92,'A  Deployment setup: prediction and uncertainty both cross a domain boundary',ha='center',fontsize=11,fontweight='bold',transform=ax.transAxes)
ax1=fig.add_subplot(gs[1,0]); x=np.arange(3); width=.35
ax1.bar(x-width/2,source_acc*100,width,label='Source test'); ax1.bar(x+width/2,field_acc*100,width,label='Field target')
ax1.set_xticks(x); ax1.set_xticklabels(tasks,rotation=20,ha='right',fontsize=8); ax1.set_ylabel('Accuracy (%)'); ax1.set_ylim(0,105); ax1.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.28),ncol=2); ax1.set_title('B  Predictive transfer failure',loc='left',fontweight='bold',fontsize=10)
for i,v in enumerate(field_acc): ax1.text(i+width/2,v*100+2,f'{v*100:.1f}',ha='center',fontsize=7)
ax2=fig.add_subplot(gs[1,1]); ax2.bar(x,source_cov*100,width=.55); ax2.axhline(90,ls='--',lw=1,label='90% nominal')
ax2.set_xticks(x); ax2.set_xticklabels(tasks,rotation=20,ha='right',fontsize=8); ax2.set_ylabel('Empirical coverage (%)'); ax2.set_ylim(0,100); ax2.legend(frameon=False,fontsize=8); ax2.set_title('C  Source-calibrated conformal sets fail in field',loc='left',fontweight='bold',fontsize=10)
for i,v in enumerate(source_cov): ax2.text(i,v*100+2,f'{v*100:.1f}',ha='center',fontsize=7)
fig.savefig(OUT/'Fig1_transfer_failure.pdf',bbox_inches='tight'); fig.savefig(OUT/'Fig1_transfer_failure.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Figure 3
bud=[0,10,20,40,80,160]
series={'Corn->PlantDoc':pdconf['aggregate']['0.1'],'Corn->PlantWild':pwconf['aggregate']['0.1'],'Apple->PlantDoc':apple['conformal']['aggregate']}
rows=[]
for task,d in series.items():
    for b in bud:
        z=d[str(b)] if str(b) in d else d[b]; rows.append([task,b,z['coverage_mean'],z['avg_set_size_mean'],z['full_set_rate_mean']])
with open(TAB/'fig3_conformal_recovery_source.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['task','target_calibration_labels','coverage','avg_set_size','full_set_rate']); w.writerows(rows)
fig,axs=plt.subplots(1,3,figsize=(7.3,2.75),sharex=True)
metrics=[('coverage_mean','Coverage',1.0),('avg_set_size_mean','Average set size',3.0),('full_set_rate_mean','Full-set rate',1.0)]
for ax,(key,label,ymax) in zip(axs,metrics):
    for task,d in series.items():
        vals=[]
        for b in bud:
            z=d[str(b)] if str(b) in d else d[b]; vals.append(z[key])
        ax.plot(bud,vals,marker='o',ms=3,label=task)
    ax.set_xscale('symlog',linthresh=10); ax.set_xlabel('Target calibration labels'); ax.set_ylabel(label); ax.set_ylim(0,ymax*1.05); ax.grid(alpha=.2)
axs[0].axhline(.90,ls='--',lw=1); axs[0].text(12,.915,'90% nominal',fontsize=7)
axs[0].set_title('A  Marginal coverage',loc='left',fontweight='bold',fontsize=9); axs[1].set_title('B  Set efficiency',loc='left',fontweight='bold',fontsize=9); axs[2].set_title('C  Uninformative full sets',loc='left',fontweight='bold',fontsize=9)
handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(0.5,1.08)); fig.tight_layout(); fig.savefig(OUT/'Fig3_conformal_recovery.pdf',bbox_inches='tight'); fig.savefig(OUT/'Fig3_conformal_recovery.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# Figure 5
budgets=[20,40,80,160]; target_keys=['corn_plantdoc','corn_plantwild','apple_plantdoc']; target_names={'corn_plantdoc':'Corn->PlantDoc','corn_plantwild':'Corn->PlantWild','apple_plantdoc':'Apple->PlantDoc'}
rows=[]
for t in target_keys:
    for b in budgets:
        for cal in [10,19]:
            z=p11[t]['aggregate'][str(b)][str(cal)]; rows.append([target_names[t],b,cal,z['adapt_n'],z['accuracy_mean'],z['coverage_mean'],z['avg_set_size_mean']])
with open(TAB/'fig5_validity_budget_source.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['task','B','n_cal','n_adapt','accuracy','coverage','avg_set_size']); w.writerows(rows)
fig=plt.figure(figsize=(7.3,5.6),constrained_layout=True); gs=fig.add_gridspec(2,2,height_ratios=[0.85,1.15])
for idx,(title,order) in enumerate([('A  Adapt-first can distort calibration',['Active\nadaptation','Calibration from\nleftovers','Held-out\nfield']),('B  Calibration-first partition',['Random calibration\nreserve','Active adaptation\non remainder','Held-out\nfield'])]):
    ax=fig.add_subplot(gs[0,idx]); ax.axis('off'); ax.set_title(title,loc='left',fontweight='bold',fontsize=10); xs=[.01,.345,.68]; box_w=.30
    for x,text in zip(xs,order):
        p=FancyBboxPatch((x,.36),box_w,.28,boxstyle='round,pad=.015',transform=ax.transAxes,fill=False,lw=1.3); p.set_clip_on(False); ax.add_patch(p); ax.text(x+box_w/2,.50,text,ha='center',va='center',fontsize=6.8,linespacing=1.15,transform=ax.transAxes)
    for x1,x2 in [(0.315,.34),(0.65,.675)]: ax.add_patch(FancyArrowPatch((x1,.5),(x2,.5),arrowstyle='-|>',mutation_scale=11,lw=1.1,transform=ax.transAxes))
    ax.text(.49,.20,'High-uncertainty cases can be removed\nbefore calibration' if idx==0 else 'Calibration sample is reserved\nbefore acquisition',ha='center',va='center',fontsize=8,transform=ax.transAxes)
ax3=fig.add_subplot(gs[1,0]); ax4=fig.add_subplot(gs[1,1])
for t in target_keys:
    name=target_names[t]; lean=[p11[t]['aggregate'][str(b)]['10']['coverage_mean'] for b in budgets]; valid=[p11[t]['aggregate'][str(b)]['19']['coverage_mean'] for b in budgets]
    ax3.plot(budgets,lean,marker='o',ls='--',label=name+' lean'); ax3.plot(budgets,valid,marker='o',ls='-',label=name+' ncal=19')
ax3.axhline(.90,ls=':',lw=1); ax3.set_ylim(.78,1.0); ax3.set_xlabel('Total field-label budget B'); ax3.set_ylabel('Pooled empirical coverage'); ax3.set_title('C  Conservative reserve raises mean coverage',loc='left',fontweight='bold',fontsize=9); ax3.grid(alpha=.2); ax3.legend(frameon=False,fontsize=6,ncol=2)
lean_frac=[max(b-10,0)/b for b in budgets]; valid_frac=[max(b-19,0)/b for b in budgets]
ax4.plot(budgets,lean_frac,marker='o',label='Lean ncal=10'); ax4.plot(budgets,valid_frac,marker='o',label='Validity-first ncal=19'); ax4.set_ylim(0,1); ax4.set_xlabel('Total field-label budget B'); ax4.set_ylabel('Fraction available for adaptation'); ax4.set_title('D  Validity has an annotation cost',loc='left',fontweight='bold',fontsize=9); ax4.grid(alpha=.2); ax4.legend(frameon=False,fontsize=8)
fig.savefig(OUT/'Fig5_calibration_first_validity.pdf',bbox_inches='tight'); fig.savefig(OUT/'Fig5_calibration_first_validity.png',dpi=300,bbox_inches='tight'); plt.close(fig)
print('done')
