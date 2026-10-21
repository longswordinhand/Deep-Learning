from pathlib import Path
import json,csv
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'journal_extension/results/phase12_publication_full_allocation_grid.json'
OUT=ROOT/'journal_extension/figures'; TAB=ROOT/'journal_extension/tables'
OUT.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)
d=json.load(open(RES))
labels={'corn_plantdoc':'Corn→PlantDoc','corn_plantwild':'Corn→PlantWild','apple_plantdoc':'Apple→PlantDoc'}
budgets=[20,40,80,160]
fracs=[0.0,.25,.5,.75,1.0]
rows=[]
for t,z in d.items():
    for B in budgets:
        for f in fracs:
            v=z['aggregate'][str(B)][str(f)]
            rows.append([labels[t],B,f,v['adapt_n'],v['cal_n'],v['accuracy_mean'],v['coverage_mean'],v['avg_set_size_mean'],v['singleton_rate_mean'],v['full_set_rate_mean']])
with open(TAB/'fig4_pareto_source.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['target','budget','adapt_frac','n_adapt','n_cal','accuracy','coverage','avg_set_size','singleton_rate','full_set_rate']); w.writerows(rows)

# 3-panel Pareto plot. Color = total budget; marker size = prediction-set size.
fig,axs=plt.subplots(1,3,figsize=(11.2,3.8),sharey=True,constrained_layout=True)
markers={0.0:'o',.25:'s',.5:'D',.75:'^',1.0:'X'}
for ax,(t,z) in zip(axs,d.items()):
    for B in budgets:
        xs=[]; ys=[]; ss=[]
        for f in fracs:
            v=z['aggregate'][str(B)][str(f)]
            xs.append(v['accuracy_mean']); ys.append(v['coverage_mean']); ss.append(35+70*v['avg_set_size_mean']/3.0)
        ax.plot(xs,ys,linewidth=1,alpha=.8,label=f'B={B}')
        for f,x,y,s in zip(fracs,xs,ys,ss):
            ax.scatter(x,y,s=s,marker=markers[f],edgecolor='black',linewidth=.4,zorder=3)
    ax.axhline(.90,color='black',linestyle='--',linewidth=.9)
    ax.set_title(labels[t]); ax.set_xlabel('Pooled field accuracy')
    ax.grid(alpha=.18,linewidth=.5)
axs[0].set_ylabel('Pooled marginal coverage')
# budget legend on first panel; fraction legend below whole figure
axs[0].legend(title='Total label budget',fontsize=7,title_fontsize=7,frameon=False,loc='lower left')
handles=[plt.Line2D([0],[0],marker=markers[f],linestyle='None',markerfacecolor='none',markeredgecolor='black',label=f'{int(f*100)}% adapt') for f in fracs]
fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,fontsize=7,bbox_to_anchor=(.5,-.04))
size_handles=[plt.scatter([],[],s=35+70*sv/3.0,facecolor='gray',edgecolor='black',linewidth=.4,label=f'set size {sv}') for sv in [1,2,3]]
fig.legend(handles=size_handles,loc='lower center',ncol=3,frameon=False,fontsize=7,bbox_to_anchor=(.5,-.13),title='Marker size = avg. prediction-set size',title_fontsize=7)
for i,ax in enumerate(axs): ax.text(-.12,1.06,chr(65+i),transform=ax.transAxes,fontweight='bold',fontsize=12)
fig.savefig(OUT/'Fig4_allocation_pareto.pdf',bbox_inches='tight')
fig.savefig(OUT/'Fig4_allocation_pareto.png',dpi=300,bbox_inches='tight')
plt.close(fig)

# Table 3: representative pooled allocation modes at B=20 and B=160, all three targets.
# Show endpoints + midpoint + 75%-adapt operating point, enough to expose the tradeoff without a 60-row table.
with open(TAB/'Table3_allocation_mechanism.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['Target','B','Adapt fraction','n_adapt','n_cal','Accuracy','Coverage','Avg set size','Singleton rate','Full-set rate'])
    for t,z in d.items():
        for B in [20,160]:
            for f in [0.0,.5,.75,1.0]:
                v=z['aggregate'][str(B)][str(f)]
                w.writerow([labels[t],B,f,v['adapt_n'],v['cal_n'],v['accuracy_mean'],v['coverage_mean'],v['avg_set_size_mean'],v['singleton_rate_mean'],v['full_set_rate_mean']])
print('WROTE',OUT/'Fig4_allocation_pareto.pdf')
print('WROTE',TAB/'Table3_allocation_mechanism.csv')
