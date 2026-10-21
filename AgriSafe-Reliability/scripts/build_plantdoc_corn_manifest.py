"""Build data/processed/plantdoc_corn_manifest.csv from features/dinov2_small_field.npz.

Mirrors the PlantWild manifest convention (path,label,original_split,sha256,primary_clean).
Exclusion rule (kill-audit fix B1, 2026-10-21):
- a byte-identical (sha256) group whose members carry conflicting labels -> ALL members primary_clean=0
  (the true label is unknowable; keeping either copy injects a coin-flip label);
- a byte-identical group with one label -> keep the first row, remaining copies primary_clean=0.
"""
import csv, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
f = np.load(ROOT / 'features/dinov2_small_field.npz')
ys = f['y']; paths = f['paths'].astype(str)
rows = []
for y, p in zip(ys, paths):
    sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    split = Path(p).parts[Path(p).parts.index('plantdoc') + 1] if 'plantdoc' in Path(p).parts else ''
    rows.append({'path': p, 'label': int(y), 'original_split': split, 'sha256': sha, 'primary_clean': 1})
groups = defaultdict(list)
for i, r in enumerate(rows): groups[r['sha256']].append(i)
for sha, idxs in groups.items():
    if len(idxs) < 2: continue
    labels = {rows[i]['label'] for i in idxs}
    if len(labels) > 1:
        for i in idxs: rows[i]['primary_clean'] = 0
    else:
        for i in idxs[1:]: rows[i]['primary_clean'] = 0
out = ROOT / 'data/processed/plantdoc_corn_manifest.csv'
with out.open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['path', 'label', 'original_split', 'sha256', 'primary_clean'])
    w.writeheader(); w.writerows(rows)
excl = [r for r in rows if r['primary_clean'] == 0]
print('rows', len(rows), 'excluded', len(excl), 'kept', len(rows) - len(excl))
for r in excl: print('EXCLUDED label', r['label'], r['path'])
print('WROTE', out)
