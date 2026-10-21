from __future__ import annotations
import csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parents[1]
ALPHA=0.10; BUDGETS=[20,40,80,160]; CALS=[0,10,20,40]; REPEATS=50

def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5
def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
def qhat(scores):
 n=len(scores)
 if n==0:return 1.0
 level=min(1.0,math.ceil((n+1)*(1-ALPHA))/n)
 return float(np.quantile(scores,level,method='higher'))
def fit(Xs,ys,C,Xt=None,yt=None):
 if Xt is None or len(Xt)==0:return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(Xs,ys)
 ns,nt=len(Xs),len(Xt); n=ns+nt; X=np.r_[Xs,Xt]; y=np.r_[ys,yt]; sw=np.r_[np.full(ns,n/(2*ns)),np.full(nt,n/(2*nt))]
 return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(X,y,sample_weight=sw)
def ev(p,y,q):
 pred=p.argmax(1); sets=p>=(1-q); size=sets.sum(1)
 return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'coverage':float(sets[np.arange(len(y)),y].mean()),'avg_set_size':float(size.mean()),'singleton_rate':float((size==1).mean()),'full_set_rate':float((size==p.shape[1]).mean())}
def load(target):
 if target.startswith('corn_'):
  s=np.load(ROOT/'features/dinov2_small_source.npz'); Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); sm={}
  with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as fh:
   for r in csv.DictReader(fh):sm[str((ROOT/r['path']).resolve())]=r['split']
  sp=np.array([sm[str((ROOT/z).resolve())] for z in ps]); tr=sp=='train'; scal=sp=='cal'
  if target=='corn_plantdoc':
   f=np.load(ROOT/'features/dinov2_small_field.npz'); Xt=normalize(f['X']); yt=f['y']; pt=f['paths'].astype(str)
   keep={}
   with (ROOT/'data/processed/plantdoc_corn_manifest.csv').open() as fh:
       for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
   m=np.array([keep.get(str(z),0)==1 for z in pt]); Xt=Xt[m]; yt=yt[m]; pt=pt[m]
  else:
   f=np.load(ROOT/'features/dinov2_small_plantwild_corn.npz'); X=normalize(f['X']); y=f['y']; p=f['paths'].astype(str); keep={}
   with (ROOT/'data/processed/plantwild_corn_manifest.csv').open() as fh:
    for r in csv.DictReader(fh):keep[r['path']]=int(r['primary_clean'])
   m=np.array([keep.get(str(z),0)==1 for z in p]); Xt=X[m]; yt=y[m]; pt=p[m]
  return Xs[tr],ys[tr],Xs[scal],ys[scal],Xt,yt,pt,30.0
 s=np.load(ROOT/'features/dinov2_small_apple_source.npz'); Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); sm={}
 with (ROOT/'data/processed/apple_source_split_seedhash.csv').open() as fh:
  for r in csv.DictReader(fh):sm[r['path']]=r['split']
 sp=np.array([sm[str(z)] for z in ps]); tr=sp=='train'; scal=sp=='cal'; f=np.load(ROOT/'features/dinov2_small_apple_plantdoc.npz'); Xt=normalize(f['X']); yt=f['y']; pt=f['paths'].astype(str)
 return Xs[tr],ys[tr],Xs[scal],ys[scal],Xt,yt,pt,30.0

def run(target):
 Xs,ys,Xcal,ycal,Xt,yt,pt,C=load(target); folds=np.array([fold_of(z) for z in pt]); base=fit(Xs,ys,C); be=entropy(base.predict_proba(Xt))
 configs=[(B,cn) for B in BUDGETS for cn in CALS if cn<=B]
 def prep(B,cn,fold):
  an=B-cn; hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]; adapt=pool[np.argsort(-be[pool])[:min(an,len(pool))]] if an else np.array([],int); clf=fit(Xs,ys,C,Xt[adapt],yt[adapt]) if len(adapt) else base; rem=np.setdiff1d(pool,adapt,assume_unique=False); hp=clf.predict_proba(Xt[hold]); return B,cn,fold,an,hold,rem,clf,hp
 preps=Parallel(n_jobs=8,backend='loky')(delayed(prep)(B,cn,fold) for B,cn in configs for fold in range(5)); cache={(B,cn,fold):(an,hold,rem,clf,hp) for B,cn,fold,an,hold,rem,clf,hp in preps}
 rows=[]
 for B,cn in configs:
  reps=1 if cn==0 else REPEATS
  for rep in range(reps):
   fr=[]
   for fold in range(5):
    an,hold,rem,clf,hp=cache[(B,cn,fold)]
    if cn>0:
     rng=np.random.default_rng(20260914+100000*rep+1000*B+10*cn+fold); ci=rng.choice(rem,size=min(cn,len(rem)),replace=False); cp=clf.predict_proba(Xt[ci]); q=qhat(1-cp[np.arange(len(ci)),yt[ci]])
    else:
     cp=clf.predict_proba(Xcal); q=qhat(1-cp[np.arange(len(ycal)),ycal])
    fr.append(ev(hp,yt[hold],q))
   keys=fr[0].keys(); rows.append({'budget':B,'cal_n':cn,'adapt_n':B-cn,'repeat':rep,**{k:float(np.mean([x[k] for x in fr])) for k in keys}})
 agg={}
 for B in BUDGETS:
  agg[str(B)]={}
  for cn in [x for x in CALS if x<=B]:
   rr=[r for r in rows if r['budget']==B and r['cal_n']==cn]; z={'adapt_n':B-cn,'cal_n':cn}
   for k in ['accuracy','macro_f1','coverage','avg_set_size','singleton_rate','full_set_rate']:
    v=np.array([r[k] for r in rr]); z[k+'_mean']=float(v.mean()); z[k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
   agg[str(B)][str(cn)]=z
 return {'target':target,'alpha':ALPHA,'aggregate':agg,'records':rows}

def main():
 out={}
 for t in ['corn_plantdoc','corn_plantwild','apple_plantdoc']:
  print('RUN',t,flush=True); out[t]=run(t)
 p=ROOT/'journal_extension/results/phase7_car_budget_rule.json'; p.write_text(json.dumps(out,indent=2)); print('WROTE',p)
if __name__=='__main__':main()
