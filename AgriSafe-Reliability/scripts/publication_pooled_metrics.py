from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
ALPHA=0.10
BUDGETS=[20,40,80,160]
CALS=[10,19]
REPEATS=30
TARGETS=['corn_plantdoc','corn_plantwild','apple_plantdoc']

def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5
def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
def qhat(s):
    n=len(s); level=min(1.0, math.ceil((n+1)*(1-ALPHA))/n)
    return float(np.quantile(s, level, method='higher'))
def fit(Xs,ys,C,Xt=None,yt=None):
    if Xt is None or len(Xt)==0:
        return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(Xs,ys)
    ns,nt=len(Xs),len(Xt); n=ns+nt
    X=np.r_[Xs,Xt]; y=np.r_[ys,yt]
    sw=np.r_[np.full(ns,n/(2*ns)),np.full(nt,n/(2*nt))]
    return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(X,y,sample_weight=sw)
def load(t):
    if t.startswith('corn_'):
        s=np.load(ROOT/'features/dinov2_small_source.npz'); Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); sm={}
        with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as fh:
            for r in csv.DictReader(fh): sm[str((ROOT/r['path']).resolve())]=r['split']
        sp=np.array([sm[str((ROOT/z).resolve())] for z in ps]); tr=sp=='train'
        if t=='corn_plantdoc':
            f=np.load(ROOT/'features/dinov2_small_field.npz'); Xt=normalize(f['X']); yt=f['y']; pt=f['paths'].astype(str)
        else:
            f=np.load(ROOT/'features/dinov2_small_plantwild_corn.npz'); X=normalize(f['X']); y=f['y']; p=f['paths'].astype(str); keep={}
            with (ROOT/'data/processed/plantwild_corn_manifest.csv').open() as fh:
                for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
            m=np.array([keep.get(str(z),0)==1 for z in p]); Xt=X[m]; yt=y[m]; pt=p[m]
        return Xs[tr],ys[tr],Xt,yt,pt,30.0
    s=np.load(ROOT/'features/dinov2_small_apple_source.npz'); Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); sm={}
    with (ROOT/'data/processed/apple_source_split_seedhash.csv').open() as fh:
        for r in csv.DictReader(fh): sm[r['path']]=r['split']
    sp=np.array([sm[str(z)] for z in ps]); tr=sp=='train'
    f=np.load(ROOT/'features/dinov2_small_apple_plantdoc.npz')
    return Xs[tr],ys[tr],normalize(f['X']),f['y'],f['paths'].astype(str),30.0

def pooled_metrics(probs,y,sets):
    pred=probs.argmax(1); sz=sets.sum(1)
    return {
        'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro')),
        'coverage':float(sets[np.arange(len(y)),y].mean()),
        'avg_set_size':float(sz.mean()),
        'singleton_rate':float((sz==1).mean()),
        'full_set_rate':float((sz==probs.shape[1]).mean())
    }

def run_target(t):
    Xs,ys,Xt,yt,pt,C=load(t); folds=np.array([fold_of(z) for z in pt]); base=fit(Xs,ys,C); be=entropy(base.predict_proba(Xt))
    n=len(yt); k=len(np.unique(yt))
    def one(B,cal_n,rep):
        probs=np.zeros((n,k),dtype=float); sets=np.zeros((n,k),dtype=bool)
        adapt_n=B-cal_n
        for fold in range(5):
            hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]
            rng=np.random.default_rng(20260914+100000*rep+1000*B+10*cal_n+fold)
            cal=rng.choice(pool,size=min(cal_n,len(pool)),replace=False)
            adapt_pool=np.setdiff1d(pool,cal,assume_unique=False)
            adapt=adapt_pool[np.argsort(-be[adapt_pool])[:min(adapt_n,len(adapt_pool))]] if adapt_n>0 else np.array([],int)
            clf=fit(Xs,ys,C,Xt[adapt],yt[adapt]) if len(adapt) else base
            cp=clf.predict_proba(Xt[cal]); qq=qhat(1-cp[np.arange(len(cal)),yt[cal]])
            hp=clf.predict_proba(Xt[hold]); probs[hold]=hp; sets[hold]=hp>=(1-qq)
        return {'target':t,'budget':B,'cal_n':cal_n,'adapt_n':adapt_n,'repeat':rep,**pooled_metrics(probs,yt,sets)}
    rows=Parallel(n_jobs=8,backend='loky',verbose=5)(delayed(one)(B,c,r) for B in BUDGETS for c in CALS if c<=B for r in range(REPEATS))
    agg={}
    for B in BUDGETS:
        agg[str(B)]={}
        for c in CALS:
            if c>B: continue
            rr=[x for x in rows if x['budget']==B and x['cal_n']==c]
            z={'cal_n':c,'adapt_n':B-c,'repeats':len(rr)}
            for key in ['accuracy','macro_f1','coverage','avg_set_size','singleton_rate','full_set_rate']:
                v=np.array([x[key] for x in rr],float)
                z[key+'_mean']=float(v.mean()); z[key+'_sd']=float(v.std(ddof=1)); z[key+'_ci95']=[float(np.quantile(v,.025)),float(np.quantile(v,.975))]
            agg[str(B)][str(c)]=z
    return {'target':t,'protocol':'publication pooled held-out metrics; calibration-first random reserve; entropy adaptation from remaining pool; 5-fold cross-fit; exact global metrics after concatenating held-out folds','alpha':ALPHA,'aggregate':agg,'records':rows}

def main():
    out={}
    for t in TARGETS:
        print('RUN',t,flush=True); out[t]=run_target(t)
    p=ROOT/'journal_extension/results/phase11_publication_pooled_metrics.json'; p.write_text(json.dumps(out,indent=2))
    print('WROTE',p)
    for t,z in out.items():
        print('\n'+t)
        for B,opts in z['aggregate'].items():
            for c,v in opts.items():
                print('B',B,'cal',c,'acc',round(v['accuracy_mean'],4),'cov',round(v['coverage_mean'],4),'size',round(v['avg_set_size_mean'],3),'covCI',tuple(round(x,4) for x in v['coverage_ci95']))
if __name__=='__main__': main()
