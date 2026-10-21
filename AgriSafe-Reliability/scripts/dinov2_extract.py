from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_mapping, plantdoc_samples

DEFAULT_MODEL_ID = 'facebook/dinov2-small'

class PathDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows; self.processor = processor
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, y = self.rows[i]
        with Image.open(ROOT / p) as im:
            img = im.convert('RGB')
        px = self.processor(images=img, return_tensors='pt')['pixel_values'][0]
        return px, int(y), str(p)

def read_source(split=None):
    rows=[]
    with (ROOT/'data/processed/corn_source_split_seed42.csv').open() as f:
        r=csv.DictReader(f)
        for row in r:
            if split is None or row['split']==split:
                rows.append((row['path'], int(row['label'])))
    return rows

def extract(model, loader, device):
    feats=[]; ys=[]; paths=[]
    model.eval()
    with torch.no_grad():
        for px,y,p in loader:
            px=px.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                out=model(pixel_values=px)
                z=out.last_hidden_state[:,0,:]
            feats.append(z.float().cpu().numpy()); ys.extend(y.numpy().tolist()); paths.extend(p)
    return np.concatenate(feats), np.asarray(ys,dtype=np.int64), np.asarray(paths)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-id',default=DEFAULT_MODEL_ID)
    ap.add_argument('--stem',default='dinov2_small')
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(args.model_id)
    model=AutoModel.from_pretrained(args.model_id).to(device)
    mapping=load_mapping(ROOT/'configs/corn_mapping.json')
    source=read_source(None)
    field=plantdoc_samples(ROOT/'data/raw/plantdoc',mapping,('train','test'))
    if args.smoke:
        source=source[:32]; field=field[:32]
    datasets={'source':source,'field':field}
    outdir=ROOT/'features'; outdir.mkdir(exist_ok=True)
    summary={'model_id':args.model_id,'stem':args.stem,'device':str(device),'smoke':args.smoke,'sets':{}}
    t0=time.time()
    for name,rows in datasets.items():
        loader=DataLoader(PathDataset(rows,proc),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
        X,y,p=extract(model,loader,device)
        suffix='_smoke' if args.smoke else ''
        np.savez_compressed(outdir/f'{args.stem}_{name}{suffix}.npz',X=X,y=y,paths=p)
        summary['sets'][name]={'n':int(len(y)),'dim':int(X.shape[1])}
        print(name, X.shape)
    summary['elapsed_sec']=time.time()-t0
    result_name=f'{args.stem}_extract_smoke.json' if args.smoke else f'{args.stem}_extract.json'
    (ROOT/'results'/result_name).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
