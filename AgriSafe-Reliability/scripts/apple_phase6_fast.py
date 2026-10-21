from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from apple_third_domain import load, fit, met, choose, entropy, qhat, conf_eval, CGRID

Xs,ys,sp,Xt,yt,pt,folds=load(); tr=sp=='train'; va=sp=='val'; cal=sp=='cal'; te=sp=='test'
T=[]
for C in CGRID:
    clf=fit(C,Xs[tr],ys[tr]); m=met(clf.predict_proba(Xs[va]),ys[va]); T.append({'C':C,**m})
best=max(T,key=lambda z:(z['macro_f1'],z['accuracy']))['C']
base=fit(best,Xs[tr],ys[tr]); bp=base.predict_proba(Xt)
source_test=met(base.predict_proba(Xs[te]),ys[te]); baseline=met(bp,yt)
print('BASE',best,source_test['accuracy'],baseline['accuracy'],baseline['ece15'],flush=True)

BUD=[0,3,6,15,30,60]
def act(kind,b,rep):
    P=[];Y=[]
    for fold in range(5):
        hold=np.where(folds==fold)[0]; cand=np.where(folds!=fold)[0]
        rng=np.random.default_rng(20260914+100000*rep+1000*b+fold)
        sel=cand[choose(kind,Xt[cand],bp[cand],b,rng)]
        clf=fit(best,Xs[tr],ys[tr],Xt[sel],yt[sel]) if len(sel) else base
        P.append(clf.predict_proba(Xt[hold]));Y.append(yt[hold])
    return {'strategy':kind,'budget':b,'repeat':rep,**met(np.vstack(P),np.concatenate(Y))}

cases=[('random',b,r) for b in BUD for r in range(20)] + [(k,b,0) for k in ['entropy','least_confidence','margin','diversity'] for b in BUD if b]
rows=Parallel(n_jobs=16,backend='loky')(delayed(act)(*c) for c in cases)
active={}
for kind in ['random','entropy','least_confidence','margin','diversity']:
    active[kind]={}
    for b in BUD:
        rr=[r for r in rows if r['strategy']==kind and r['budget']==b]
        if not rr: continue
        active[kind][str(b)]={}
        for k in ['accuracy','macro_f1','nll','ece15']:
            v=np.array([z[k] for z in rr]);active[kind][str(b)][k+'_mean']=float(v.mean());active[kind][str(b)][k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
print('ACTIVE_DONE',flush=True)

srcp=base.predict_proba(Xs[cal]); srcq=qhat(1-srcp[np.arange(cal.sum()),ys[cal]],0.1); CB=[0,10,20,40,80,160]
def conf(b,rep):
    V=[]; rng=np.random.default_rng(20260915+100000*rep+1000*b)
    for fold in range(5):
        hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]
        if b==0:q=srcq
        else:
            s=rng.choice(pool,size=min(b,len(pool)),replace=False);q=qhat(1-bp[s,yt[s]],0.1)
        V.append(conf_eval(bp[hold],yt[hold],q))
    return {'budget':b,'repeat':rep,**{k:float(np.mean([z[k] for z in V])) for k in V[0]}}

crows=Parallel(n_jobs=16,backend='loky')(delayed(conf)(b,r) for b in CB for r in range(50 if b else 1))
conformal={}
for b in CB:
    rr=[r for r in crows if r['budget']==b]; conformal[str(b)]={}
    for k in ['coverage','avg_set_size','singleton_rate','full_set_rate']:
        v=np.array([z[k] for z in rr]);conformal[str(b)][k+'_mean']=float(v.mean());conformal[str(b)][k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
print('CONFORMAL_DONE',flush=True)

JB=[20,40,80,160]; FR=[0,0.25,0.5,0.75,1.0]
cache={}
for B in JB:
    for frac in FR:
        na=int(round(B*frac)); nc=B-na
        for fold in range(5):
            hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]
            aidx=pool[np.argsort(-entropy(bp[pool]))[:min(na,len(pool))]] if na else np.array([],int)
            clf=fit(best,Xs[tr],ys[tr],Xt[aidx],yt[aidx]) if len(aidx) else base
            cache[(B,frac,fold)]={'hold':hold,'pool':pool,'adapt':aidx,'ph':clf.predict_proba(Xt[hold]),'poolp':clf.predict_proba(Xt[pool])}
print('JOINT_MODELS_DONE',flush=True)

def joint(B,frac,rep):
    na=int(round(B*frac)); nc=B-na; V=[]
    for fold in range(5):
        d=cache[(B,frac,fold)]; hold=d['hold']; pool=d['pool']; adapt=d['adapt']; ph=d['ph']
        rng=np.random.default_rng(20260916+100000*rep+1000*B+100*int(frac*100)+fold)
        remain=np.setdiff1d(pool,adapt,assume_unique=False)
        if nc:
            csel=rng.choice(remain,size=min(nc,len(remain)),replace=False)
            pos={x:i for i,x in enumerate(pool)}; inds=np.array([pos[x] for x in csel])
            q=qhat(1-d['poolp'][inds,yt[csel]],0.1)
        else:q=srcq
        ce=conf_eval(ph,yt[hold],q)
        V.append({'accuracy':float((ph.argmax(1)==yt[hold]).mean()),'macro_f1':float(f1_score(yt[hold],ph.argmax(1),average='macro')),**ce})
    return {'B':B,'frac':frac,'adapt_n':na,'cal_n':nc,'repeat':rep,**{k:float(np.mean([z[k] for z in V])) for k in V[0]}}

jrows=Parallel(n_jobs=16,backend='loky')(delayed(joint)(B,f,r) for B in JB for f in FR for r in range(20))
jointagg={}
for B in JB:
    jointagg[str(B)]={}
    for f in FR:
        rr=[z for z in jrows if z['B']==B and z['frac']==f]; x={'adapt_n':rr[0]['adapt_n'],'cal_n':rr[0]['cal_n']}
        for k in ['accuracy','macro_f1','coverage','avg_set_size','singleton_rate','full_set_rate']:
            v=np.array([z[k] for z in rr]);x[k+'_mean']=float(v.mean());x[k+'_sd']=float(v.std(ddof=1))
        jointagg[str(B)][str(f)]=x
out={'task':'Apple PlantVillage->PlantDoc','selected_C':best,'source_tuning':T,'source_test':source_test,'field_baseline':baseline,'active':active,'conformal':{'alpha':0.1,'source_qhat':srcq,'aggregate':conformal},'joint':{'alpha':0.1,'aggregate':jointagg},'protocol':'source-only C selection; DINOv2-small frozen; 5-fold path-hash target cross-fit; target labels revealed only after acquisition or for disjoint target calibration.'}
p=ROOT/'journal_extension/results/phase6_apple_third_domain.json';p.write_text(json.dumps(out,indent=2));print('WROTE',p,flush=True)
