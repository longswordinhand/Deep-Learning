from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
ROOT=Path(__file__).resolve().parents[1]
MODEL='facebook/dinov2-small'
class DS(Dataset):
    def __init__(self,rows,proc): self.rows=rows; self.proc=proc
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        p,y=self.rows[i]
        with Image.open(ROOT/p) as im: img=im.convert('RGB')
        px=self.proc(images=img,return_tensors='pt')['pixel_values'][0]
        return px,int(y),str(p)
def extract(rows,proc,model,device):
    loader=DataLoader(DS(rows,proc),batch_size=128,shuffle=False,num_workers=8,pin_memory=True)
    X=[]; y=[]; p=[]
    model.eval()
    with torch.no_grad():
        for px,yy,pp in loader:
            px=px.to(device,non_blocking=True)
            with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                z=model(pixel_values=px).last_hidden_state[:,0,:]
            X.append(z.float().cpu().numpy()); y.extend(yy.numpy().tolist()); p.extend(pp)
    return np.concatenate(X),np.asarray(y,dtype=np.int64),np.asarray(p)
def main():
    src=[]
    with (ROOT/'data/processed/apple_source_split_seedhash.csv').open() as f:
        for r in csv.DictReader(f): src.append((r['path'],int(r['label'])))
    tgt=[]
    with (ROOT/'data/processed/apple_plantdoc_manifest.csv').open() as f:
        for r in csv.DictReader(f):
            if int(r['primary_clean'])==1: tgt.append((r['path'],int(r['label'])))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL); model=AutoModel.from_pretrained(MODEL).to(device)
    out={}; t=time.time()
    for name,rows in [('apple_source',src),('apple_plantdoc',tgt)]:
        X,y,p=extract(rows,proc,model,device); np.savez_compressed(ROOT/f'features/dinov2_small_{name}.npz',X=X,y=y,paths=p)
        out[name]={'n':len(y),'dim':X.shape[1]}; print(name,X.shape)
    out['elapsed_sec']=time.time()-t
    (ROOT/'journal_extension/results/phase6_apple_extract.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
