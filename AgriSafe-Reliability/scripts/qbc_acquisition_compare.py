from __future__ import annotations
import argparse, csv, hashlib, json, math, os
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
C=30.0
COMMITTEE_N=9
COMMITTEE_FRAC=0.80
COMMITTEE_SEED=20260914

def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5

def ece(p,y,bins=15):
    conf=p.max(1); pred=p.argmax(1); ok=(pred==y).astype(float); edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        m=(conf>=edges[i])&((conf<edges[i+1]) if i<bins-1 else (conf<=edges[i+1]))
        if m.any(): out+=m.mean()*abs(ok[m].mean()-conf[m].mean())
    return float(out)

def met(p,y):
    pred=p.argmax(1); conf=p.max(1)
    out={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),
         'nll':float(log_loss(y,p,labels=[0,1,2])),'ece15':ece(p,y)}
    order=np.argsort(-conf)
    for cov in [0.5,0.7,0.8]:
        k=max(1,int(math.ceil(cov*len(y)))); a=float((pred[order[:k]]==y[order[:k]]).mean())
        out[f'acc_at_{int(cov*100)}cov']=a; out[f'risk_at_{int(cov*100)}cov']=1-a
    return out

def fit(Xs,ys,Xt=None,yt=None):
    if Xt is None or len(Xt)==0:
        return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(Xs,ys)
    ns,nt=len(Xs),len(Xt); n=ns+nt; X=np.r_[Xs,Xt]; y=np.r_[ys,yt]
    sw=np.r_[np.full(ns,n/(2*ns)),np.full(nt,n/(2*nt))]
    return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(X,y,sample_weight=sw)

def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)

def build_committee(Xs,ys,n=COMMITTEE_N,frac=COMMITTEE_FRAC):
    members=[]
    classes=np.unique(ys)
    for m in range(n):
        rng=np.random.default_rng(COMMITTEE_SEED+m)
        idx=[]
        for c in classes:
            ci=np.where(ys==c)[0]; k=max(2,int(round(frac*len(ci))))
            idx.extend(rng.choice(ci,size=k,replace=True).tolist())
        idx=np.asarray(idx,int)
        clf=LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=COMMITTEE_SEED+m).fit(Xs[idx],ys[idx])
        members.append(clf)
    return members

def committee_probs(members,X):
    # shape M,N,K
    return np.stack([m.predict_proba(X) for m in members],axis=0)

def qbc_scores(cp):
    meanp=cp.mean(0)
    pred_ent=entropy(meanp)
    mi=pred_ent - np.mean(np.stack([entropy(cp[m]) for m in range(cp.shape[0])],axis=0),axis=0)
    votes=np.argmax(cp,axis=2) # M,N
    vote_p=np.zeros((cp.shape[1],cp.shape[2]),float)
    for k in range(cp.shape[2]): vote_p[:,k]=(votes==k).mean(0)
    vote_ent=entropy(vote_p)
    return {'committee_predictive_entropy':pred_ent,'committee_mi':mi,'vote_entropy':vote_ent},meanp

def source_train():
    s=np.load(ROOT/'features/dinov2_small_source.npz'); X=normalize(s['X']); y=s['y']; p=s['paths'].astype(str)
    sm={}
    with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as fh:
        for r in csv.DictReader(fh): sm[str((ROOT/r['path']).resolve())]=r['split']
    tr=np.array([sm[str((ROOT/z).resolve())]=='train' for z in p])
    return X[tr],y[tr]

def target_load(name):
    if name=='plantdoc':
        f=np.load(ROOT/'features/dinov2_small_field.npz'); X=normalize(f['X']); y=f['y']; p=f['paths'].astype(str)
        keep={}
        with (ROOT/'data/processed/plantdoc_corn_manifest.csv').open() as fh:
            for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
        m=np.array([keep.get(str(z),0)==1 for z in p]); return X[m],y[m],p[m]
    f=np.load(ROOT/'features/dinov2_small_plantwild_corn.npz'); X=normalize(f['X']); y=f['y']; p=f['paths'].astype(str)
    # primary-clean manifest: keep==1, keyed by repo-relative path
    keep={}
    with (ROOT/'data/processed/plantwild_corn_manifest.csv').open() as fh:
        for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
    m=np.array([keep.get(str(z),0)==1 for z in p])
    return X[m],y[m],p[m]

def score_quality(score,base_p,y):
    err=(base_p.argmax(1)!=y).astype(int)
    if len(np.unique(err))<2:return float('nan')
    return float(roc_auc_score(err,score))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target',choices=['plantdoc','plantwild'],required=True); ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args(); budgets=[0,3] if a.smoke else [0,3,6,15,30,60]; random_repeats=2 if a.smoke else 20; jobs=2 if a.smoke else 8
    Xs,ys=source_train(); Xt,yt,pt=target_load(a.target); folds=np.array([fold_of(z) for z in pt])
    base=fit(Xs,ys); base_p=base.predict_proba(Xt); members=build_committee(Xs,ys); cp=committee_probs(members,Xt); qscore,committee_mean=qbc_scores(cp)
    scores={'single_entropy':entropy(base_p),**qscore}
    score_audit={k:{'error_auroc':score_quality(v,base_p,yt),'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1))} for k,v in scores.items()}

    def one(kind,b,rep):
        pp=[]; yy=[]; cc=[]
        for fold in range(5):
            hold=np.where(folds==fold)[0]; cand=np.where(folds!=fold)[0]
            if b==0: sel=np.array([],int)
            elif kind=='random':
                rng=np.random.default_rng(20260914+100000*rep+1000*b+fold); sel=rng.choice(cand,min(b,len(cand)),replace=False)
            else:
                sel=cand[np.argsort(-scores[kind][cand])[:min(b,len(cand))]]
            clf=fit(Xs,ys,Xt[sel],yt[sel]) if len(sel) else base
            pp.append(clf.predict_proba(Xt[hold])); yy.append(yt[hold]); cc.append([int((yt[sel]==c).sum()) for c in [0,1,2]])
        return {'strategy':kind,'budget':b,'repeat':rep,**met(np.vstack(pp),np.concatenate(yy)),'selected_class_counts_by_fold':cc}

    cases=[]
    for b in budgets:
        for r in range(random_repeats): cases.append(('random',b,r))
    for kind in ['single_entropy','committee_predictive_entropy','committee_mi','vote_entropy']:
        for b in budgets:
            if b>0: cases.append((kind,b,0))
    rows=Parallel(n_jobs=jobs,backend='loky',verbose=5)(delayed(one)(*x) for x in cases)
    agg={}
    for kind in ['random','single_entropy','committee_predictive_entropy','committee_mi','vote_entropy']:
        agg[kind]={}
        for b in budgets:
            rr=[r for r in rows if r['strategy']==kind and r['budget']==b]
            if not rr: continue
            agg[kind][str(b)]={}
            for k in ['accuracy','macro_f1','nll','ece15','acc_at_50cov','acc_at_70cov','acc_at_80cov']:
                v=np.array([r[k] for r in rr]); agg[kind][str(b)][k+'_mean']=float(v.mean()); agg[kind][str(b)][k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
    out={'target':a.target,'protocol':'same frozen DINOv2 features and path-hash 5-fold evaluation; committee is source-only stratified bootstrap logistic probes; acquisition never sees candidate target labels','committee':{'n':COMMITTEE_N,'source_fraction':COMMITTEE_FRAC,'seed':COMMITTEE_SEED},'baseline':met(base_p,yt),'committee_mean_metrics':met(committee_mean,yt),'score_audit':score_audit,'aggregate':agg,'records':rows}
    suffix='_smoke' if a.smoke else ''
    p=ROOT/'journal_extension/results'/f'phase3_qbc_{a.target}{suffix}.json'; p.write_text(json.dumps(out,indent=2))
    print(json.dumps({'target':a.target,'baseline':out['baseline'],'score_audit':score_audit,'aggregate':agg},indent=2)); print('WROTE',p)
if __name__=='__main__': main()
