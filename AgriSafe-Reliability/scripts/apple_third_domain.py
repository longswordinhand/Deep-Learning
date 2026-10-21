from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parents[1]
CGRID=[0.01,0.03,0.1,0.3,1,3,10,30]
BUDGETS=[0,3,6,15,30,60]
ALPHAS=[0.10]

def fold_of(p): return int(hashlib.sha256(str(p).encode()).hexdigest()[:8],16)%5
def entropy(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
def ece(p,y,bins=15):
    conf=p.max(1); pred=p.argmax(1); ok=(pred==y).astype(float); edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        m=(conf>=edges[i])&((conf<edges[i+1]) if i<bins-1 else (conf<=edges[i+1]))
        if m.any(): out+=m.mean()*abs(ok[m].mean()-conf[m].mean())
    return float(out)
def met(p,y):
    pred=p.argmax(1); conf=p.max(1); out={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'nll':float(log_loss(y,p,labels=[0,1,2])),'ece15':ece(p,y),'mean_confidence':float(conf.mean())}
    order=np.argsort(-conf)
    for cov in [0.5,0.7,0.8]:
        k=max(1,int(math.ceil(cov*len(y)))); out[f'acc_at_{int(cov*100)}cov']=float((pred[order[:k]]==y[order[:k]]).mean())
    return out
def qhat(scores,alpha):
    n=len(scores)
    if n==0:return 1.0
    level=min(1.0,math.ceil((n+1)*(1-alpha))/n)
    return float(np.quantile(scores,level,method='higher'))
def conf_eval(p,y,q):
    sets=p>=(1-q); size=sets.sum(1)
    return {'coverage':float(sets[np.arange(len(y)),y].mean()),'avg_set_size':float(size.mean()),'singleton_rate':float((size==1).mean()),'full_set_rate':float((size==3).mean())}
def fit(C,Xs,ys,Xt=None,yt=None):
    if Xt is None or len(Xt)==0:return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(Xs,ys)
    ns,nt=len(Xs),len(Xt); n=ns+nt; sw=np.r_[np.full(ns,n/(2*ns)),np.full(nt,n/(2*nt))]
    return LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(np.r_[Xs,Xt],np.r_[ys,yt],sample_weight=sw)
def choose(kind,X,p,b,rng):
    b=min(b,len(X))
    if b==0:return np.array([],int)
    if kind=='random':return rng.choice(len(X),b,replace=False)
    if kind=='entropy':return np.argsort(-entropy(p))[:b]
    if kind=='least_confidence':return np.argsort(p.max(1))[:b]
    if kind=='margin':
        sp=np.sort(p,axis=1); return np.argsort(sp[:,-1]-sp[:,-2])[:b]
    # k-center diversity
    mu=normalize(X.mean(0,keepdims=True))[0]; sel=[int(np.argmax(X@mu))]; md=1-X@X[sel[0]]
    while len(sel)<b:
        md[sel]=-np.inf; j=int(np.argmax(md)); sel.append(j); md=np.minimum(md,1-X@X[j])
    return np.array(sel)
def load():
    s=np.load(ROOT/'features/dinov2_small_apple_source.npz'); t=np.load(ROOT/'features/dinov2_small_apple_plantdoc.npz')
    Xs=normalize(s['X']); ys=s['y']; ps=s['paths'].astype(str); Xt=normalize(t['X']); yt=t['y']; pt=t['paths'].astype(str)
    sm={}
    with (ROOT/'data/processed/apple_source_split_seedhash.csv').open() as f:
        for r in csv.DictReader(f): sm[r['path']]=r['split']
    sp=np.array([sm[p] for p in ps]); folds=np.array([fold_of(p) for p in pt])
    return Xs,ys,sp,Xt,yt,pt,folds
def main():
    Xs,ys,sp,Xt,yt,pt,folds=load(); tr=sp=='train'; va=sp=='val'; cal=sp=='cal'; te=sp=='test'
    # source-only C selection
    tune=[]
    for C in CGRID:
        clf=fit(C,Xs[tr],ys[tr]); p=clf.predict_proba(Xs[va]); m=met(p,ys[va]); tune.append({'C':C,**m})
    best=max(tune,key=lambda z:(z['macro_f1'],z['accuracy']))['C']
    base=fit(best,Xs[tr],ys[tr]); bp=base.predict_proba(Xt); source_test=met(base.predict_proba(Xs[te]),ys[te]); baseline=met(bp,yt)
    print('selected_C',best,'source_test',source_test['accuracy'],source_test['macro_f1'],'field',baseline['accuracy'],baseline['macro_f1'],baseline['ece15'])
    # active acquisition
    def active_one(kind,b,rep):
        pp=[]; yy=[]
        for fold in range(5):
            hold=np.where(folds==fold)[0]; cand=np.where(folds!=fold)[0]; rng=np.random.default_rng(20260914+100000*rep+1000*b+fold)
            loc=choose(kind,Xt[cand],bp[cand],b,rng); sel=cand[loc]; clf=fit(best,Xs[tr],ys[tr],Xt[sel],yt[sel]) if len(sel) else base
            pp.append(clf.predict_proba(Xt[hold])); yy.append(yt[hold])
        return {'strategy':kind,'budget':b,'repeat':rep,**met(np.vstack(pp),np.concatenate(yy))}
    cases=[]
    for b in BUDGETS:
        for r in range(20):cases.append(('random',b,r))
    for kind in ['entropy','least_confidence','margin','diversity']:
        for b in BUDGETS:
            if b>0:cases.append((kind,b,0))
    rows=Parallel(n_jobs=8,backend='loky')(delayed(active_one)(*c) for c in cases)
    active={}
    for kind in ['random','entropy','least_confidence','margin','diversity']:
        active[kind]={}
        for b in BUDGETS:
            rr=[r for r in rows if r['strategy']==kind and r['budget']==b]
            if not rr:continue
            active[kind][str(b)]={}
            for k in ['accuracy','macro_f1','nll','ece15','acc_at_50cov','acc_at_70cov','acc_at_80cov']:
                v=np.array([z[k] for z in rr]);active[kind][str(b)][k+'_mean']=float(v.mean());active[kind][str(b)][k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
    # conformal recovery, source calibration vs random target calibration
    srcp=base.predict_proba(Xs[cal]); src_scores=1-srcp[np.arange(cal.sum()),ys[cal]]; src_q=qhat(src_scores,0.1)
    conf_budgets=[0,10,20,40,80,160]
    def conf_one(b,rep):
        vals=[]; rng=np.random.default_rng(20260915+100000*rep+1000*b)
        for fold in range(5):
            hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]
            if b==0:q=src_q
            else:
                sel=rng.choice(pool,size=min(b,len(pool)),replace=False); q=qhat(1-bp[sel,yt[sel]],0.1)
            vals.append(conf_eval(bp[hold],yt[hold],q))
        return {'budget':b,'repeat':rep,**{k:float(np.mean([z[k] for z in vals])) for k in vals[0]}}
    ccases=[(b,r) for b in conf_budgets for r in range(50 if b>0 else 1)]
    crows=Parallel(n_jobs=8,backend='loky')(delayed(conf_one)(*c) for c in ccases)
    conformal={}
    for b in conf_budgets:
        rr=[z for z in crows if z['budget']==b];conformal[str(b)]={}
        for k in ['coverage','avg_set_size','singleton_rate','full_set_rate']:
            v=np.array([z[k] for z in rr]);conformal[str(b)][k+'_mean']=float(v.mean());conformal[str(b)][k+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.0
    # joint allocation with entropy-selected adaptation + disjoint random target calibration
    JB=[20,40,80,160]; FR=[0,0.25,0.5,0.75,1.0]
    def joint_one(B,frac,rep):
        na=int(round(B*frac)); nc=B-na; vals=[]
        for fold in range(5):
            hold=np.where(folds==fold)[0]; pool=np.where(folds!=fold)[0]
            ent=entropy(bp[pool]); adapt=pool[np.argsort(-ent)[:min(na,len(pool))]] if na else np.array([],int)
            remain=np.setdiff1d(pool,adapt,assume_unique=False); rng=np.random.default_rng(20260916+100000*rep+1000*B+100*int(frac*100)+fold)
            csel=rng.choice(remain,size=min(nc,len(remain)),replace=False) if nc else np.array([],int)
            clf=fit(best,Xs[tr],ys[tr],Xt[adapt],yt[adapt]) if len(adapt) else base
            ph=clf.predict_proba(Xt[hold]); acc=float((ph.argmax(1)==yt[hold]).mean()); mf=float(f1_score(yt[hold],ph.argmax(1),average='macro'))
            if nc:q=qhat(1-clf.predict_proba(Xt[csel])[np.arange(len(csel)),yt[csel]],0.1)
            else:q=src_q
            ce=conf_eval(ph,yt[hold],q); vals.append({'accuracy':acc,'macro_f1':mf,**ce})
        keys=vals[0].keys();return {'budget':B,'frac':frac,'repeat':rep,'adapt_n':na,'cal_n':nc,**{k:float(np.mean([z[k] for z in vals])) for k in keys}}
    jcases=[(B,f,r) for B in JB for f in FR for r in range(20)]
    jrows=Parallel(n_jobs=8,backend='loky')(delayed(joint_one)(*c) for c in jcases)
    joint={}
    for B in JB:
        joint[str(B)]={}
        for f in FR:
            rr=[z for z in jrows if z['budget']==B and z['frac']==f]; x={'adapt_n':rr[0]['adapt_n'],'cal_n':rr[0]['cal_n']}
            for k in ['accuracy','macro_f1','coverage','avg_set_size','singleton_rate','full_set_rate']:
                v=np.array([z[k] for z in rr]);x[k+'_mean']=float(v.mean());x[k+'_sd']=float(v.std(ddof=1))
            joint[str(B)][str(f)]=x
    out={'task':'apple PlantVillage->PlantDoc 3-class','mapping':{'healthy':0,'apple_scab':1,'cedar_apple_rust':2},'selected_C':best,'source_tuning':tune,'source_test':source_test,'field_baseline':baseline,'active':active,'conformal':{'alpha':0.1,'source_qhat':src_q,'aggregate':conformal},'joint':{'alpha':0.1,'aggregate':joint},'protocol':'DINOv2-small frozen features; source-only hyperparameter selection; 5-fold path-hash target cross-fit; target labels revealed only after acquisition or for disjoint calibration.'}
    p=ROOT/'journal_extension/results/phase6_apple_third_domain.json';p.write_text(json.dumps(out,indent=2));print('WROTE',p)
    print(json.dumps({'selected_C':best,'source_test':source_test,'baseline':baseline,'active':active,'conformal':conformal,'joint':joint},indent=2))
if __name__=='__main__':main()
