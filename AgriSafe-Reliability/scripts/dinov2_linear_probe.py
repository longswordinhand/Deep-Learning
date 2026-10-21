from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]

def ece_score(probs, labels, bins=15):
    conf=probs.max(1); pred=probs.argmax(1); edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        lo,hi=edges[i],edges[i+1]
        m=(conf>lo)&(conf<=hi) if i else (conf>=lo)&(conf<=hi)
        if m.any(): out += m.mean()*abs((pred[m]==labels[m]).mean()-conf[m].mean())
    return float(out)

def metrics(y, probs):
    pred=probs.argmax(1)
    return {
        'n':int(len(y)),
        'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro')),
        'nll':float(log_loss(y,probs,labels=[0,1,2])),
        'ece15':ece_score(probs,y),
        'mean_confidence':float(probs.max(1).mean()),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist()
    }

def load_split_map():
    m={}
    with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as f:
        r=csv.DictReader(f)
        for row in r: m[str(ROOT/row['path'])]=row['split']
    return m

def main():
    src=np.load(ROOT/'features/dinov2_small_source.npz', allow_pickle=False)
    fld=np.load(ROOT/'features/dinov2_small_field.npz', allow_pickle=False)
    Xs=normalize(src['X']); ys=src['y']; ps=np.array(src['paths']).astype(str)
    Xf=normalize(fld['X']); yf=fld['y']; pf=np.array(fld['paths']).astype(str)
    split_map=load_split_map()
    splits=np.array([split_map[str(ROOT/p)] if not str(p).startswith(str(ROOT)) else split_map[str(p)] for p in ps])
    tr=splits=='train'; va=splits=='val'; ca=splits=='cal'; te=splits=='test'
    clf=LogisticRegression(C=1.0, max_iter=5000, solver='lbfgs', random_state=42)
    clf.fit(Xs[tr], ys[tr])
    out={'model':'facebook/dinov2-small','representation':'frozen CLS embedding, L2 normalized','classifier':'LogisticRegression C=1.0 lbfgs','train_n':int(tr.sum())}
    probs={}
    for name,mask in [('source_val',va),('source_cal',ca),('source_test',te)]:
        p=clf.predict_proba(Xs[mask]); probs[name]=p; out[name]=metrics(ys[mask],p)
    pfield=clf.predict_proba(Xf); probs['field']=pfield; out['field']=metrics(yf,pfield)
    outdir=ROOT/'results'
    (outdir/'dinov2_linear_probe.json').write_text(json.dumps(out,indent=2))
    np.savez_compressed(outdir/'dinov2_linear_probe_probs.npz',source_test_probs=probs['source_test'],source_test_labels=ys[te],field_probs=pfield,field_labels=yf,field_paths=pf)
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
