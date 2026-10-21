from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
C=30.0
ALPHAS=[0.10,0.05]
BUDGETS=[0,10,20,40,80,160]
REPEATS=50

def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5

def source_parts():
    s=np.load(ROOT/'features/dinov2_small_source.npz'); X=normalize(s['X']); y=s['y']; p=s['paths'].astype(str)
    sm={}
    with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as fh:
        for r in csv.DictReader(fh): sm[str((ROOT/r['path']).resolve())]=r['split']
    sp=np.array([sm[str((ROOT/z).resolve())] for z in p])
    return X,y,sp

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

def qhat(scores,alpha):
    n=len(scores)
    if n==0: return 1.0
    level=min(1.0, math.ceil((n+1)*(1-alpha))/n)
    return float(np.quantile(scores,level,method='higher'))

def evaluate(p,y,q):
    sets=p >= (1.0-q)
    cov=float(sets[np.arange(len(y)),y].mean())
    size=sets.sum(1)
    return {'coverage':cov,'avg_set_size':float(size.mean()),'singleton_rate':float((size==1).mean()),'full_set_rate':float((size==p.shape[1]).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target',choices=['plantdoc','plantwild'],required=True); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    budgets=[0,10,40] if a.smoke else BUDGETS; repeats=5 if a.smoke else REPEATS; jobs=2 if a.smoke else 8
    Xs,ys,sp=source_parts(); tr=sp=='train'; cal=sp=='cal'
    clf=LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(Xs[tr],ys[tr])
    srcp=clf.predict_proba(Xs[cal]); src_scores=1-srcp[np.arange(cal.sum()),ys[cal]]
    Xt,yt,pt=target_load(a.target); tp=clf.predict_proba(Xt); folds=np.array([fold_of(z) for z in pt])
    source_q={str(alpha):qhat(src_scores,alpha) for alpha in ALPHAS}

    def one(b,rep,alpha):
        fold_rows=[]
        rng=np.random.default_rng(20260914+100000*rep+1000*b+int(alpha*100))
        for fold in range(5):
            hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]
            if b==0:
                q=source_q[str(alpha)]; cal_n=int(cal.sum()); cal_kind='source_only'
            else:
                sel=rng.choice(pool,size=min(b,len(pool)),replace=False)
                scores=1-tp[sel,yt[sel]]; q=qhat(scores,alpha); cal_n=len(sel); cal_kind='target_only'
            ev=evaluate(tp[hold],yt[hold],q); fold_rows.append({'fold':fold,'qhat':q,'cal_n':cal_n,**ev})
        return {'budget':b,'repeat':rep,'alpha':alpha,'calibration':'source_only' if b==0 else 'target_only',
                'coverage':float(np.mean([z['coverage'] for z in fold_rows])),
                'avg_set_size':float(np.mean([z['avg_set_size'] for z in fold_rows])),
                'singleton_rate':float(np.mean([z['singleton_rate'] for z in fold_rows])),
                'full_set_rate':float(np.mean([z['full_set_rate'] for z in fold_rows])),
                'qhat_mean':float(np.mean([z['qhat'] for z in fold_rows])),'folds':fold_rows}
    cases=[(b,r,a0) for a0 in ALPHAS for b in budgets for r in range(repeats if b>0 else 1)]
    rows=Parallel(n_jobs=jobs,backend='loky',verbose=5)(delayed(one)(*c) for c in cases)
    agg={}
    for alpha in ALPHAS:
        aa={}
        for b in budgets:
            rr=[r for r in rows if r['alpha']==alpha and r['budget']==b]
            aa[str(b)]={}
            for k in ['coverage','avg_set_size','singleton_rate','full_set_rate','qhat_mean']:
                v=np.array([r[k] for r in rr]); aa[str(b)][k+'_mean']=float(v.mean()); aa[str(b)][k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
        agg[str(alpha)]=aa
    out={'target':a.target,'protocol':'source-trained DINOv2-small linear probe fixed; 5-fold target cross-fit; random target calibration labels from non-held-out folds only; target labels used only for conformal calibration; primary-clean PlantWild','budgets':budgets,'repeats':repeats,'source_calibration_n':int(cal.sum()),'source_qhat':source_q,'aggregate':agg,'records':rows}
    suffix='_smoke' if a.smoke else ''
    p=ROOT/'journal_extension/results'/f'phase4_conformal_recovery_{a.target}{suffix}.json'; p.write_text(json.dumps(out,indent=2))
    print(json.dumps({'target':a.target,'source_qhat':source_q,'aggregate':agg},indent=2)); print('WROTE',p)
if __name__=='__main__': main()
