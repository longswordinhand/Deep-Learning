from __future__ import annotations
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parents[1]
ALPHA=.10; BUDGETS=[20,40,80,160]; CAL_N=19; REPEATS=30
TARGETS=['corn_plantdoc','corn_plantwild','apple_plantdoc']
def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5
def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
def qhat(s):
 n=len(s); level=min(1.,math.ceil((n+1)*(1-ALPHA))/n); return float(np.quantile(s,level,method='higher'))
def fit(Xs,ys,C,Xt=None,yt=None):
 if Xt is None or len(Xt)==0:return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(Xs,ys)
 ns,nt=len(Xs),len(Xt); n=ns+nt; X=np.r_[Xs,Xt]; y=np.r_[ys,yt]; sw=np.r_[np.full(ns,n/(2*ns)),np.full(nt,n/(2*nt))]
 return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(X,y,sample_weight=sw)
def ev(p,y,q):
 pred=p.argmax(1); sets=p>=(1-q); sz=sets.sum(1)
 return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'coverage':float(sets[np.arange(len(y)),y].mean()),'avg_set_size':float(sz.mean()),'singleton_rate':float((sz==1).mean()),'full_set_rate':float((sz==p.shape[1]).mean())}
def load(t):
 if t.startswith('corn_'):
  s=np.load(ROOT/'features/dinov2_small_source.npz'); Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); sm={}
  with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as fh:
   for r in csv.DictReader(fh):sm[str((ROOT/r['path']).resolve())]=r['split']
  sp=np.array([sm[str((ROOT/z).resolve())] for z in ps]); tr=sp=='train'
  if t=='corn_plantdoc':
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
  return Xs[tr],ys[tr],Xt,yt,pt,30.
 s=np.load(ROOT/'features/dinov2_small_apple_source.npz'); Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); sm={}
 with (ROOT/'data/processed/apple_source_split_seedhash.csv').open() as fh:
  for r in csv.DictReader(fh):sm[r['path']]=r['split']
 sp=np.array([sm[str(z)] for z in ps]); tr=sp=='train'; f=np.load(ROOT/'features/dinov2_small_apple_plantdoc.npz'); return Xs[tr],ys[tr],normalize(f['X']),f['y'],f['paths'].astype(str),30.
def run_target(t):
 Xs,ys,Xt,yt,pt,C=load(t); folds=np.array([fold_of(z) for z in pt]); base=fit(Xs,ys,C); be=entropy(base.predict_proba(Xt))
 def one(B,rep):
  fr=[]; an=B-CAL_N
  for fold in range(5):
   hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]
   rng=np.random.default_rng(20260914+100000*rep+1000*B+fold)
   # Critical order: random calibration reserve FIRST, before any active acquisition.
   cal=rng.choice(pool,size=min(CAL_N,len(pool)),replace=False)
   adapt_pool=np.setdiff1d(pool,cal,assume_unique=False)
   adapt=adapt_pool[np.argsort(-be[adapt_pool])[:min(an,len(adapt_pool))]] if an>0 else np.array([],int)
   clf=fit(Xs,ys,C,Xt[adapt],yt[adapt]) if len(adapt) else base
   cp=clf.predict_proba(Xt[cal]); qq=qhat(1-cp[np.arange(len(cal)),yt[cal]])
   fr.append(ev(clf.predict_proba(Xt[hold]),yt[hold],qq))
  return {'budget':B,'cal_n':CAL_N,'adapt_n':an,'repeat':rep,**{k:float(np.mean([x[k] for x in fr])) for k in fr[0]}}
 rows=Parallel(n_jobs=8,backend='loky')(delayed(one)(B,r) for B in BUDGETS for r in range(REPEATS))
 agg={}
 for B in BUDGETS:
  rr=[x for x in rows if x['budget']==B]; z={'cal_n':CAL_N,'adapt_n':B-CAL_N}
  for k in ['accuracy','macro_f1','coverage','avg_set_size','singleton_rate','full_set_rate']:
   v=np.array([x[k] for x in rr]); z[k+'_mean']=float(v.mean()); z[k+'_sd']=float(v.std(ddof=1))
  agg[str(B)]=z
 return {'target':t,'protocol':'validity-floor calibration-first partition: n_cal=ceil((2-alpha)/alpha)=19 at alpha=0.1; random calibration reserve selected before entropy-based adaptation acquisition; disjoint 5-fold cross-fit','alpha':ALPHA,'cal_n':CAL_N,'repeats':REPEATS,'aggregate':agg,'records':rows}
def main():
 out={}
 for t in TARGETS:
  print('RUN',t,flush=True); out[t]=run_target(t)
 p=ROOT/'journal_extension/results/phase10_validity_floor19.json';p.write_text(json.dumps(out,indent=2));print(json.dumps({t:z['aggregate'] for t,z in out.items()},indent=2));print('WROTE',p)
if __name__=='__main__':main()
