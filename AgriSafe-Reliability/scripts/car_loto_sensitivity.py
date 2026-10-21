from __future__ import annotations
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parents[1]
ALPHA=.10; BUDGETS=[20,40,80,160]; RHOS=[.05,.075,.10,.125,.15,.20,.25]; REPEATS=50
TARGETS=['corn_plantdoc','corn_plantwild','apple_plantdoc']
def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5
def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
def qhat(s):
 n=len(s); level=min(1.0,math.ceil((n+1)*(1-ALPHA))/n); return float(np.quantile(s,level,method='higher'))
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
   for r in csv.DictReader(fh): sm[str((ROOT/r['path']).resolve())]=r['split']
  sp=np.array([sm[str((ROOT/z).resolve())] for z in ps]); tr=sp=='train'; ca=sp=='cal'
  if t=='corn_plantdoc':
   f=np.load(ROOT/'features/dinov2_small_field.npz'); Xt=normalize(f['X']); yt=f['y']; pt=f['paths'].astype(str)
   keep={}
   with (ROOT/'data/processed/plantdoc_corn_manifest.csv').open() as fh:
       for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
   m=np.array([keep.get(str(z),0)==1 for z in pt]); Xt=Xt[m]; yt=yt[m]; pt=pt[m]
  else:
   f=np.load(ROOT/'features/dinov2_small_plantwild_corn.npz'); X=normalize(f['X']); y=f['y']; p=f['paths'].astype(str); keep={}
   with (ROOT/'data/processed/plantwild_corn_manifest.csv').open() as fh:
    for r in csv.DictReader(fh): keep[r['path']]=int(r['primary_clean'])
   m=np.array([keep.get(str(z),0)==1 for z in p]); Xt=X[m]; yt=y[m]; pt=p[m]
  return Xs[tr],ys[tr],Xs[ca],ys[ca],Xt,yt,pt,30.
 s=np.load(ROOT/'features/dinov2_small_apple_source.npz'); Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); sm={}
 with (ROOT/'data/processed/apple_source_split_seedhash.csv').open() as fh:
  for r in csv.DictReader(fh):sm[r['path']]=r['split']
 sp=np.array([sm[str(z)] for z in ps]); tr=sp=='train'; ca=sp=='cal'; f=np.load(ROOT/'features/dinov2_small_apple_plantdoc.npz'); Xt=normalize(f['X']); yt=f['y']; pt=f['paths'].astype(str)
 return Xs[tr],ys[tr],Xs[ca],ys[ca],Xt,yt,pt,30.
def run_target(t):
 Xs,ys,Xcal,ycal,Xt,yt,pt,C=load(t); folds=np.array([fold_of(z) for z in pt]); base=fit(Xs,ys,C); be=entropy(base.predict_proba(Xt))
 configs=[]
 for B in BUDGETS:
  for rho in RHOS:
   cn=min(B,max(math.ceil(1/ALPHA),math.ceil(rho*B))); configs.append((B,rho,cn))
 uniq=sorted(set((B,cn) for B,rho,cn in configs))
 def prep(B,cn,fold):
  an=B-cn; hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]; adapt=pool[np.argsort(-be[pool])[:an]] if an else np.array([],int); clf=fit(Xs,ys,C,Xt[adapt],yt[adapt]) if len(adapt) else base; rem=np.setdiff1d(pool,adapt); return B,cn,fold,hold,rem,clf,clf.predict_proba(Xt[hold])
 pre=Parallel(n_jobs=8,backend='loky')(delayed(prep)(B,cn,f) for B,cn in uniq for f in range(5)); cache={(B,cn,f):(h,r,c,p) for B,cn,f,h,r,c,p in pre}
 rows=[]
 for B,rho,cn in configs:
  for rep in range(REPEATS):
   fr=[]
   for fold in range(5):
    hold,rem,clf,hp=cache[(B,cn,fold)]; rng=np.random.default_rng(20260914+100000*rep+1000*B+10*cn+fold); ci=rng.choice(rem,size=cn,replace=False); cp=clf.predict_proba(Xt[ci]); qq=qhat(1-cp[np.arange(cn),yt[ci]]); fr.append(ev(hp,yt[hold],qq))
   rows.append({'budget':B,'rho':rho,'cal_n':cn,'adapt_n':B-cn,'repeat':rep,**{k:float(np.mean([x[k] for x in fr])) for k in fr[0]}})
 agg={}
 for rho in RHOS:
  agg[str(rho)]={}
  for B in BUDGETS:
   rr=[r for r in rows if r['rho']==rho and r['budget']==B]; z={'cal_n':rr[0]['cal_n'],'adapt_n':rr[0]['adapt_n']}
   for k in ['accuracy','macro_f1','coverage','avg_set_size','singleton_rate','full_set_rate']:
    v=np.array([r[k] for r in rr]); z[k+'_mean']=float(v.mean()); z[k+'_sd']=float(v.std(ddof=1))
   agg[str(rho)][str(B)]=z
 return {'target':t,'aggregate':agg,'records':rows}
def select_rho(results,train_targets):
 feasible=[]
 for rho in RHOS:
  cells=[results[t]['aggregate'][str(rho)][str(B)] for t in train_targets for B in BUDGETS]
  mincov=min(x['coverage_mean'] for x in cells); meanacc=float(np.mean([x['accuracy_mean'] for x in cells])); meansz=float(np.mean([x['avg_set_size_mean'] for x in cells]))
  if mincov>=.90: feasible.append((meanacc,-meansz,-rho,rho,mincov))
 if feasible:return max(feasible)[3]
 # fail-closed fallback: maximize worst coverage, then accuracy
 cand=[]
 for rho in RHOS:
  cells=[results[t]['aggregate'][str(rho)][str(B)] for t in train_targets for B in BUDGETS]; cand.append((min(x['coverage_mean'] for x in cells),np.mean([x['accuracy_mean'] for x in cells]),-rho,rho))
 return max(cand)[3]
def main():
 results={}
 for t in TARGETS:
  print('RUN',t,flush=True); results[t]=run_target(t)
 loto={}
 for held in TARGETS:
  train=[t for t in TARGETS if t!=held]; rho=select_rho(results,train); cells=results[held]['aggregate'][str(rho)]; loto[held]={'train_targets':train,'selected_rho':rho,'heldout':cells,'heldout_all_coverage_ge_090':all(cells[str(B)]['coverage_mean']>=.90 for B in BUDGETS),'heldout_mean_accuracy':float(np.mean([cells[str(B)]['accuracy_mean'] for B in BUDGETS]))}
 out={'alpha':ALPHA,'rhos':RHOS,'budgets':BUDGETS,'selection_rule':'using training tasks only: among rho with mean coverage >=0.90 in every train-task x budget cell, maximize mean accuracy; tie-break lower mean set size then lower rho; fail-closed fallback maximizes worst coverage','results':results,'leave_one_task_out':loto}
 p=ROOT/'journal_extension/results/phase8_car_loto_sensitivity.json'; p.write_text(json.dumps(out,indent=2)); print(json.dumps(loto,indent=2)); print('WROTE',p)
if __name__=='__main__':main()
