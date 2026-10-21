from pathlib import Path
import json, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'journal_extension/figures'; OUT.mkdir(parents=True,exist_ok=True)
TAB=ROOT/'journal_extension/tables'; TAB.mkdir(parents=True,exist_ok=True)

# ---- Figure 2 source data ----
common_methods=['entropy','least_confidence','margin','diversity']
budgets=[3,6,15,30,60]
targets=[('Corn→PlantDoc','phase1_active_label_budget.json'),('Corn→PlantWild','phase2_plantwild_active_label_budget.json')]
rows=[]
for tname,fn in targets:
    d=json.load(open(ROOT/'journal_extension/results'/fn)); agg=d['aggregate']; r=agg['random']
    for b in budgets:
        rv=r[str(b)]['accuracy_mean']
        for m in common_methods:
            key='uncertainty' if m=='entropy' else m
            if str(b) in agg[key]:
                v=agg[key][str(b)]['accuracy_mean']
                rows.append((tname,b,m,100*(v-rv)))
# apple
a=json.load(open(ROOT/'journal_extension/results/phase6_apple_third_domain.json'))['active']
for b in budgets:
    rv=a['random'][str(b)]['accuracy_mean']
    for m in common_methods:
        key='entropy' if m=='entropy' else m
        v=a[key][str(b)]['accuracy_mean']
        rows.append(('Apple→PlantDoc',b,m,100*(v-rv)))

with open(TAB/'fig2_acquisition_delta_source.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['target','budget','method','accuracy_delta_pp_vs_random']); w.writerows(rows)

# QBC rows
qbc_rows=[]
for tname,fn in [('Corn→PlantDoc','phase3_qbc_plantdoc.json'),('Corn→PlantWild','phase3_qbc_plantwild.json')]:
    d=json.load(open(ROOT/'journal_extension/results'/fn))['aggregate']; r=d['random']
    for b in budgets:
        rv=r[str(b)]['accuracy_mean']
        for m in ['single_entropy','committee_predictive_entropy','committee_mi','vote_entropy']:
            v=d[m][str(b)]['accuracy_mean']
            qbc_rows.append((tname,b,m,100*(v-rv)))
with open(TAB/'fig2_qbc_delta_source.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['target','budget','method','accuracy_delta_pp_vs_random']); w.writerows(qbc_rows)

# Plot Figure 2: 4 panels, 3 standard + 1 committee summary
GLIM=float(max(8,np.ceil(max(abs(v) for _,_,_,v in rows)/2)*2))  # one shared colour scale for panels A-C
fig,axs=plt.subplots(2,2,figsize=(10.6,7.4),constrained_layout=True)
method_labels=['Entropy','Least conf.','Margin','Diversity']
for ax,tname in zip(axs.flat[:3],['Corn→PlantDoc','Corn→PlantWild','Apple→PlantDoc']):
    mat=np.array([[next(v for t,b,m,v in rows if t==tname and b==B and m==M) for B in budgets] for M in common_methods])
    lim=GLIM
    im=ax.imshow(mat,aspect='auto',cmap='coolwarm',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
    ax.set_xticks(range(len(budgets)),budgets); ax.set_xlabel('Target-label budget')
    ax.set_yticks(range(len(method_labels)),method_labels)
    ax.set_title(tname)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j,i,f'{mat[i,j]:+.1f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(mat[i,j])>0.55*lim else 'black')
    cb=fig.colorbar(im,ax=ax,shrink=.82,pad=.02); cb.set_label('Accuracy Δ vs random (pp)')
# QBC panel: two-domain lines across budgets
ax=axs.flat[3]
styles={'committee_predictive_entropy':'o-','committee_mi':'s--','vote_entropy':'^-.'}
labels={'committee_predictive_entropy':'Committee pred. entropy','committee_mi':'Committee MI','vote_entropy':'Vote entropy'}
for tname,ls in [('Corn→PlantDoc','-'),('Corn→PlantWild','--')]:
    for m,fmt in styles.items():
        vals=[next(v for t,b,mm,v in qbc_rows if t==tname and b==B and mm==m) for B in budgets]
        marker=fmt[0]
        ax.plot(budgets,vals,marker=marker,linestyle=ls,label=f'{tname}: {labels[m]}')
ax.axhline(0,linewidth=.8,color='black')
ax.set_xlabel('Target-label budget'); ax.set_ylabel('Accuracy Δ vs random (pp)')
ax.set_title('Committee disagreement is also domain-dependent')
ax.legend(fontsize=6.8,ncol=1,frameon=False)
for i,ax in enumerate(axs.flat): ax.text(-.13,1.08,chr(65+i),transform=ax.transAxes,fontweight='bold',fontsize=12)
fig.savefig(OUT/'Fig2_acquisition_instability.pdf',bbox_inches='tight')
fig.savefig(OUT/'Fig2_acquisition_instability.png',dpi=300,bbox_inches='tight')
plt.close(fig)

# ---- Table 1 ----
# Values are drawn from audited local manifests/results, not inferred from predictions.
t1=[
 ['PlantVillage corn → PlantDoc corn','Corn: gray leaf spot / northern leaf blight / rust',2690,376,370,'3 byte-identical cross-label conflict pairs (6 files) excluded after exact-file audit; no same-label duplicates','Controlled→field target'],
 ['PlantVillage corn → PlantWild corn','Corn: gray leaf spot / northern leaf blight / rust',2690,677,642,'20 exact duplicate groups; 15 cross-label conflict groups removed; same-label exact duplicates collapsed','Primary-clean PlantWild target'],
 ['PlantVillage apple → PlantDoc apple','Apple: healthy / scab / cedar rust',2550,273,273,'0 exact target duplicate groups; 0 cross-label conflicts','Independent crop task']]
with open(TAB/'Table1_data_integrity.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['Task','Crop/classes','Source n','Target raw n','Primary target n','Exact-duplicate/conflict audit','Notes']); w.writerows(t1)

# ---- Table 2: pooled publication metrics ----
p=json.load(open(ROOT/'journal_extension/results/phase11_publication_pooled_metrics.json'))
with open(TAB/'Table2_pooled_operating_points.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['Target','B','Regime','n_adapt','n_cal','Accuracy mean','Accuracy RepP2.5','Accuracy RepP97.5','Macro-F1 mean','Coverage mean','Coverage RepP2.5','Coverage RepP97.5','Avg set size','Singleton rate','Full-set rate'])
    for t,z in p.items():
        for B,opts in z['aggregate'].items():
            for c,reg in [('10','Lean'),('19','Validity-first')]:
                v=opts[c]
                w.writerow([t,int(B),reg,v['adapt_n'],v['cal_n'],v['accuracy_mean'],*v['accuracy_ci95'],v['macro_f1_mean'],v['coverage_mean'],*v['coverage_ci95'],v['avg_set_size_mean'],v['singleton_rate_mean'],v['full_set_rate_mean']])

print('WROTE',OUT/'Fig2_acquisition_instability.pdf')
print('WROTE',OUT/'Fig2_acquisition_instability.png')
print('WROTE',TAB/'Table1_data_integrity.csv')
print('WROTE',TAB/'Table2_pooled_operating_points.csv')
