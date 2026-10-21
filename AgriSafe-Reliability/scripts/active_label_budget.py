from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
C=30.0; BUDGETS=[0,3,6,15,30,60]; REPEATS=20; SEED=20260913

def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5

def ece(p,y,bins=15):
    c=p.max(1); pred=p.argmax(1); ok=(pred==y).astype(float); out=0.0
    ed=np.linspace(0,1,bins+1)
    for i in range(bins):
        m=(c>=ed[i])&((c<ed[i+1]) if i<bins-1 else (c<=ed[i+1]))
        if m.any(): out+=m.mean()*abs(ok[m].mean()-c[m].mean())
    return float(out)

def met(p,y):
    pred=p.argmax(1); conf=p.max(1); o={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'nll':float(log_loss(y,p,labels=[0,1,2])),'ece15':ece(p,y)}
    order=np.argsort(-conf)
    for cov in [0.5,0.7,0.8]:
        k=max(1,int(math.ceil(cov*len(y)))); a=float((pred[order[:k]]==y[order[:k]]).mean())
        o[f'acc_at_{int(cov*100)}cov']=a; o[f'risk_at_{int(cov*100)}cov']=1-a
    return o

def fit(Xs,ys,Xt=None,yt=None):
    if Xt is None or len(Xt)==0: return LogisticRegression(C=C,max_iter=5000,solver='lbfgs').fit(Xs,ys)
    ns,nt=len(Xs),len(Xt); n=ns+nt
    X=np.r_[Xs,Xt]; y=np.r_[ys,yt]; sw=np.r_[np.full(ns,n/(2*ns)),np.full(nt,n/(2*nt))]
    return LogisticRegression(C=C,max_iter=5000,solver='lbfgs').fit(X,y,sample_weight=sw)

def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)

def select(kind,X,p,b,cent,rng):
    n=len(X); b=min(b,n)
    if b==0:return np.array([],int)
    if kind=='random': return rng.choice(n,b,replace=False)
    if kind=='uncertainty': return np.argsort(-entropy(p))[:b]
    if kind=='diversity':
        mu=normalize(X.mean(0,keepdims=True))[0]; first=int(np.argmax(X@mu)); sel=[first]; md=1-X@X[first]
        while len(sel)<b:
            md[sel]=-np.inf; j=int(np.argmax(md)); sel.append(j); md=np.minimum(md,1-X@X[j])
        return np.array(sel)
    shift=1-(X@cent.T).max(1); sn=(shift-shift.min())/(np.ptp(shift)+1e-12); first=int(np.argmax(sn)); sel=[first]; md=1-X@X[first]
    while len(sel)<b:
        md[sel]=-np.inf; dn=np.clip(md/(np.max(md[np.isfinite(md)])+1e-12),0,1); score=(sn+1e-6)*(dn+1e-6); score[sel]=-np.inf; j=int(np.argmax(score)); sel.append(j); md=np.minimum(md,1-X@X[j])
    return np.array(sel)

def main():
    s=np.load(ROOT/'features/dinov2_small_source.npz'); f=np.load(ROOT/'features/dinov2_small_field.npz')
    Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); Xf=normalize(f['X']); yf=f['y']; pf=f['paths'].astype(str)
    keep={}
    with (ROOT/'data/processed/plantdoc_corn_manifest.csv').open() as fh:
        for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
    m=np.array([keep.get(str(z),0)==1 for z in pf]); Xf=Xf[m]; yf=yf[m]; pf=pf[m]
    sm={}
    with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as fh:
        for r in csv.DictReader(fh): sm[str((ROOT/r['path']).resolve())]=r['split']
    tr=np.array([sm[str((ROOT/p).resolve())]=='train' for p in ps]); folds=np.array([fold_of(p) for p in pf]); base=fit(Xs[tr],ys[tr]); bp=base.predict_proba(Xf)
    cent=[]
    for c in [0,1,2]:
        v=normalize(Xs[tr][ys[tr]==c].mean(0,keepdims=True))[0]; cent.append(v)
    cent=np.stack(cent)
    out={'protocol':'5-fold path-hash target evaluation; selection uses target features and source predictions only','budgets':BUDGETS,'random_repeats':REPEATS,'baseline':met(bp,yf),'records':[]}
    for kind in ['random','uncertainty','diversity','shift_diversity']:
        reps=REPEATS if kind=='random' else 1
        for b in BUDGETS:
            if b==0 and kind!='random': continue
            for rep in range(reps):
                pp=[]; yy=[]; cc=[]
                for fold in range(5):
                    hold=np.where(folds==fold)[0]; cand=np.where(folds!=fold)[0]; rng=np.random.default_rng(SEED+100000*rep+1000*b+fold)
                    loc=select(kind,Xf[cand],bp[cand],b,cent,rng); sel=cand[loc]
                    clf=fit(Xs[tr],ys[tr],Xf[sel],yf[sel]) if len(sel) else base
                    pp.append(clf.predict_proba(Xf[hold])); yy.append(yf[hold]); cc.append([int((yf[sel]==c).sum()) for c in [0,1,2]])
                rec={'strategy':kind,'budget':b,'repeat':rep,**met(np.vstack(pp),np.concatenate(yy)),'selected_class_counts_by_fold':cc}; out['records'].append(rec); print(kind,b,rep,rec['accuracy'],rec['macro_f1'],rec['ece15'],flush=True)
    agg={}
    for kind in ['random','uncertainty','diversity','shift_diversity']:
        agg[kind]={}
        for b in BUDGETS:
            rr=[r for r in out['records'] if r['strategy']==kind and r['budget']==b]
            if not rr: continue
            agg[kind][str(b)]={}
            for k in ['accuracy','macro_f1','nll','ece15','acc_at_50cov','acc_at_70cov','acc_at_80cov']:
                v=np.array([r[k] for r in rr]); agg[kind][str(b)][k+'_mean']=float(v.mean()); agg[kind][str(b)][k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
    out['aggregate']=agg
    p=ROOT/'journal_extension/results/phase1_active_label_budget.json'; p.write_text(json.dumps(out,indent=2)); print('WROTE',p)
if __name__=='__main__': main()
