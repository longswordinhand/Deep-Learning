from __future__ import annotations
import csv, hashlib, json, math, os
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
C=30.0

def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5

def ece(p,y,bins=15):
    conf=p.max(1); pred=p.argmax(1); ok=(pred==y).astype(float); edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        m=(conf>=edges[i])&((conf<edges[i+1]) if i<bins-1 else (conf<=edges[i+1]))
        if m.any(): out+=m.mean()*abs(ok[m].mean()-conf[m].mean())
    return float(out)

def met(p,y):
    pred=p.argmax(1); conf=p.max(1); out={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'nll':float(log_loss(y,p,labels=[0,1,2])),'ece15':ece(p,y)}
    order=np.argsort(-conf)
    for cov in [0.5,0.7,0.8]:
        k=max(1,int(math.ceil(cov*len(y)))); a=float((pred[order[:k]]==y[order[:k]]).mean()); out[f'acc_at_{int(cov*100)}cov']=a; out[f'risk_at_{int(cov*100)}cov']=1-a
    return out

def fit(Xs,ys,Xt=None,yt=None):
    if Xt is None or len(Xt)==0: return LogisticRegression(C=C,max_iter=5000,solver='lbfgs').fit(Xs,ys)
    ns,nt=len(Xs),len(Xt); n=ns+nt; X=np.r_[Xs,Xt]; y=np.r_[ys,yt]; sw=np.r_[np.full(ns,n/(2*ns)),np.full(nt,n/(2*nt))]
    return LogisticRegression(C=C,max_iter=5000,solver='lbfgs').fit(X,y,sample_weight=sw)

def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)

def choose(kind,X,p,b,cent,rng):
    n=len(X); b=min(b,n)
    if b==0:return np.array([],int)
    if kind=='random': return rng.choice(n,b,replace=False)
    if kind=='uncertainty': return np.argsort(-entropy(p))[:b]
    if kind=='least_confidence': return np.argsort(p.max(1))[:b]
    if kind=='margin':
        sp=np.sort(p,axis=1); return np.argsort(sp[:,-1]-sp[:,-2])[:b]
    if kind=='diversity':
        mu=normalize(X.mean(0,keepdims=True))[0]; sel=[int(np.argmax(X@mu))]; md=1-X@X[sel[0]]
        while len(sel)<b:
            md[sel]=-np.inf; j=int(np.argmax(md)); sel.append(j); md=np.minimum(md,1-X@X[j])
        return np.array(sel)
    shift=1-(X@cent.T).max(1); sn=(shift-shift.min())/(np.ptp(shift)+1e-12); sel=[int(np.argmax(sn))]; md=1-X@X[sel[0]]
    while len(sel)<b:
        md[sel]=-np.inf; dn=np.clip(md/(np.max(md[np.isfinite(md)])+1e-12),0,1); score=(sn+1e-6)*(dn+1e-6); score[sel]=-np.inf; j=int(np.argmax(score)); sel.append(j); md=np.minimum(md,1-X@X[j])
    return np.array(sel)

def load_all():
    s=np.load(ROOT/'features/dinov2_small_source.npz'); f=np.load(ROOT/'features/dinov2_small_plantwild_corn.npz')
    Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); Xf0=normalize(f['X']); yf0=f['y']; pf0=f['paths'].astype(str)
    sm={}
    with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as fh:
        for r in csv.DictReader(fh): sm[str((ROOT/r['path']).resolve())]=r['split']
    tr=np.array([sm[str((ROOT/p).resolve())]=='train' for p in ps])
    meta={}
    with (ROOT/'data/processed/plantwild_corn_manifest.csv').open() as fh:
        for r in csv.DictReader(fh): meta[r['path']]=(int(r['primary_clean']),r['sha256'])
    keep=np.array([meta[p][0]==1 for p in pf0]); Xf=Xf0[keep]; yf=yf0[keep]; pf=pf0[keep]
    folds=np.array([int(meta[p][1][:8],16)%5 for p in pf]); base=fit(Xs[tr],ys[tr]); bp=base.predict_proba(Xf)
    cent=[]
    for c in [0,1,2]: cent.append(normalize(Xs[tr][ys[tr]==c].mean(0,keepdims=True))[0])
    return Xs[tr],ys[tr],Xf,yf,folds,bp,np.stack(cent),base

def main():
    smoke=os.environ.get('AGRISAFE_SMOKE')=='1'; budgets=[0,3] if smoke else [0,3,6,15,30,60]; repeats=2 if smoke else 20; jobs=2 if smoke else 8
    Xs,ys,Xf,yf,folds,bp,cent,base=load_all()
    def one(kind,b,rep):
        pp=[]; yy=[]; cc=[]
        for fold in range(5):
            hold=np.where(folds==fold)[0]; cand=np.where(folds!=fold)[0]; rng=np.random.default_rng(20260913+100000*rep+1000*b+fold)
            loc=choose(kind,Xf[cand],bp[cand],b,cent,rng); sel=cand[loc]; clf=fit(Xs,ys,Xf[sel],yf[sel]) if len(sel) else base
            pp.append(clf.predict_proba(Xf[hold])); yy.append(yf[hold]); cc.append([int((yf[sel]==c).sum()) for c in [0,1,2]])
        return {'strategy':kind,'budget':b,'repeat':rep,**met(np.vstack(pp),np.concatenate(yy)),'selected_class_counts_by_fold':cc}
    cases=[]
    for b in budgets:
        for r in range(repeats): cases.append(('random',b,r))
    for kind in ['uncertainty','least_confidence','margin','diversity','shift_diversity']:
        for b in budgets:
            if b>0: cases.append((kind,b,0))
    rows=Parallel(n_jobs=jobs,backend='loky',verbose=5)(delayed(one)(*x) for x in cases)
    agg={}
    for kind in ['random','uncertainty','least_confidence','margin','diversity','shift_diversity']:
        agg[kind]={}
        for b in budgets:
            rr=[r for r in rows if r['strategy']==kind and r['budget']==b]
            if not rr: continue
            agg[kind][str(b)]={}
            for k in ['accuracy','macro_f1','nll','ece15','acc_at_50cov','acc_at_70cov','acc_at_80cov']:
                v=np.array([r[k] for r in rr]); agg[kind][str(b)][k+'_mean']=float(v.mean()); agg[kind][str(b)][k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
    out={'protocol':'PlantWild corn primary-clean set; 5-fold exact-hash evaluation; selection uses field embeddings and source-model predictions only; selected labels enter adaptation after acquisition','target_n':int(len(yf)),'budgets':budgets,'random_repeats':repeats,'baseline':met(bp,yf),'aggregate':agg,'records':rows}
    name='phase2_plantwild_smoke.json' if smoke else 'phase2_plantwild_active_label_budget.json'; p=ROOT/'journal_extension/results'/name; p.write_text(json.dumps(out,indent=2)); print(json.dumps(agg,indent=2)); print('WROTE',p)
if __name__=='__main__': main()
