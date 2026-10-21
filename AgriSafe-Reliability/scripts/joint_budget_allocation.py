from __future__ import annotations
import argparse,csv,hashlib,json,math,os
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
C=30.0; ALPHA=0.10
TOTAL_BUDGETS=[20,40,80,160]
ADAPT_FRACS=[0.0,0.25,0.5,0.75,1.0]
REPEATS=10

def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5

def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)

def qhat(scores,alpha=ALPHA):
    n=len(scores)
    if n==0:return 1.0
    level=min(1.0,math.ceil((n+1)*(1-alpha))/n)
    return float(np.quantile(scores,level,method='higher'))

def fit(Xs,ys,Xt=None,yt=None):
    if Xt is None or len(Xt)==0:
        return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(Xs,ys)
    ns,nt=len(Xs),len(Xt); n=ns+nt; X=np.r_[Xs,Xt]; y=np.r_[ys,yt]
    sw=np.r_[np.full(ns,n/(2*ns)),np.full(nt,n/(2*nt))]
    return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(X,y,sample_weight=sw)

def eval_all(p,y,q):
    pred=p.argmax(1); sets=p>=(1-q); size=sets.sum(1)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),
            'coverage':float(sets[np.arange(len(y)),y].mean()),'avg_set_size':float(size.mean()),
            'singleton_rate':float((size==1).mean()),'full_set_rate':float((size==p.shape[1]).mean())}

def source_parts():
    s=np.load(ROOT/'features/dinov2_small_source.npz'); X=normalize(s['X']); y=s['y']; p=s['paths'].astype(str)
    sm={}
    with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as fh:
        for r in csv.DictReader(fh): sm[str((ROOT/r['path']).resolve())]=r['split']
    sp=np.array([sm[str((ROOT/z).resolve())] for z in p]); return X,y,sp

def target_load(name):
    if name=='plantdoc':
        f=np.load(ROOT/'features/dinov2_small_field.npz'); X=normalize(f['X']); y=f['y']; p=f['paths'].astype(str)
        keep={}
        with (ROOT/'data/processed/plantdoc_corn_manifest.csv').open() as fh:
            for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
        m=np.array([keep.get(str(z),0)==1 for z in p]); return X[m],y[m],p[m]
    f=np.load(ROOT/'features/dinov2_small_plantwild_corn.npz'); X=normalize(f['X']); y=f['y']; p=f['paths'].astype(str)
    keep={}
    with (ROOT/'data/processed/plantwild_corn_manifest.csv').open() as fh:
        for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
    m=np.array([keep.get(str(z),0)==1 for z in p]); return X[m],y[m],p[m]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target',choices=['plantdoc','plantwild'],required=True); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    budgets=[20] if a.smoke else TOTAL_BUDGETS; fracs=[0.0,0.5,1.0] if a.smoke else ADAPT_FRACS; reps=2 if a.smoke else REPEATS; jobs=2 if a.smoke else 8
    Xs,ys,sp=source_parts(); tr=sp=='train'; scal=sp=='cal'; Xt,yt,pt=target_load(a.target); folds=np.array([fold_of(z) for z in pt])
    base=fit(Xs[tr],ys[tr]); base_prob=base.predict_proba(Xt); base_ent=entropy(base_prob)

    def one(B,frac,rep):
        foldres=[]
        na=int(round(B*frac)); nc=B-na
        for fold in range(5):
            hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]
            # adaptation acquisition is target-label blind: highest source-model entropy
            adapt=pool[np.argsort(-base_ent[pool])[:min(na,len(pool))]] if na>0 else np.array([],int)
            clf=fit(Xs[tr],ys[tr],Xt[adapt],yt[adapt]) if len(adapt) else base
            remaining=np.setdiff1d(pool,adapt,assume_unique=False)
            rng=np.random.default_rng(20260914+100000*rep+1000*B+100*int(frac*100)+fold)
            if nc>0:
                calidx=rng.choice(remaining,size=min(nc,len(remaining)),replace=False)
                cp=clf.predict_proba(Xt[calidx]); scores=1-cp[np.arange(len(calidx)),yt[calidx]]; q=qhat(scores)
                cal_kind='target'
            else:
                # no target labels left for calibration: use source-domain calibration only
                cp=clf.predict_proba(Xs[scal]); scores=1-cp[np.arange(scal.sum()),ys[scal]]; q=qhat(scores)
                cal_kind='source'
            hp=clf.predict_proba(Xt[hold]); ev=eval_all(hp,yt[hold],q)
            foldres.append({'fold':fold,'adapt_n':len(adapt),'cal_n':nc,'cal_kind':cal_kind,'qhat':q,**ev})
        keys=['accuracy','macro_f1','coverage','avg_set_size','singleton_rate','full_set_rate']
        return {'total_budget':B,'adapt_frac':frac,'adapt_n':na,'cal_n':nc,'repeat':rep,**{k:float(np.mean([z[k] for z in foldres])) for k in keys},'qhat_mean':float(np.mean([z['qhat'] for z in foldres])),'folds':foldres}

    cases=[(B,f,r) for B in budgets for f in fracs for r in range(reps if f<1.0 else 1)]
    rows=Parallel(n_jobs=jobs,backend='loky',verbose=5)(delayed(one)(*c) for c in cases)
    agg={}
    for B in budgets:
        agg[str(B)]={}
        for f in fracs:
            rr=[r for r in rows if r['total_budget']==B and r['adapt_frac']==f]
            x={'adapt_n':rr[0]['adapt_n'],'cal_n':rr[0]['cal_n']}
            for k in ['accuracy','macro_f1','coverage','avg_set_size','singleton_rate','full_set_rate','qhat_mean']:
                v=np.array([r[k] for r in rr]); x[k+'_mean']=float(v.mean()); x[k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
            agg[str(B)][str(f)]=x
    out={'target':a.target,'protocol':'fixed total target-label budget split into entropy-selected adaptation labels and disjoint random target conformal-calibration labels; 5-fold target cross-fit; no held-out labels used in selection/training/calibration','alpha':ALPHA,'budgets':budgets,'adapt_fracs':fracs,'repeats':reps,'aggregate':agg,'records':rows}
    suffix='_smoke' if a.smoke else ''
    p=ROOT/'journal_extension/results'/f'phase5_budget_allocation_{a.target}{suffix}.json'; p.write_text(json.dumps(out,indent=2)); print(json.dumps(agg,indent=2)); print('WROTE',p)
if __name__=='__main__':main()
