from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
CS=[0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0]

def ece_score(probs, labels, bins=15):
    conf=probs.max(1); pred=probs.argmax(1); edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        m=(conf>lo)&(conf<=hi) if i else (conf>=lo)&(conf<=hi)
        if m.any(): out += m.mean()*abs((pred[m]==labels[m]).mean()-conf[m].mean())
    return float(out)

def met(y,p):
    pred=p.argmax(1)
    return {'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'nll':float(log_loss(y,p,labels=[0,1,2])),'ece15':ece_score(p,y),'mean_confidence':float(p.max(1).mean())}

def run(feature_stem):
    src=np.load(ROOT/f'features/{feature_stem}_source.npz'); fld=np.load(ROOT/f'features/{feature_stem}_field.npz')
    Xs=normalize(src['X']); ys=src['y']; ps=np.array(src['paths']).astype(str); Xf=normalize(fld['X']); yf=fld['y']
    sm={}
    with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as f:
        for r in csv.DictReader(f): sm[str(ROOT/r['path'])]=r['split']
    splits=np.array([sm[str(ROOT/p)] if not str(p).startswith(str(ROOT)) else sm[str(p)] for p in ps])
    tr=splits=='train'; va=splits=='val'; te=splits=='test'
    search=[]
    for C in CS:
        clf=LogisticRegression(C=C,max_iter=5000,solver='lbfgs',random_state=42).fit(Xs[tr],ys[tr])
        pv=clf.predict_proba(Xs[va]); m=met(ys[va],pv); search.append({'C':C,**m})
    best=max(search,key=lambda d:(d['macro_f1'],-d['C']))
    clf=LogisticRegression(C=best['C'],max_iter=5000,solver='lbfgs',random_state=42).fit(Xs[tr],ys[tr])
    pt=clf.predict_proba(Xs[te]); pf=clf.predict_proba(Xf)
    out={'feature_stem':feature_stem,'C_grid':CS,'selection':'max source validation macro-F1; smaller C tie-break','selected_C':best['C'],'search':search,'source_test':met(ys[te],pt),'field':met(yf,pf)}
    (ROOT/'results'/f'{feature_stem}_linear_probe_tuned.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--feature-stem',default='dinov2_small'); args=ap.parse_args()
    run(args.feature_stem)
