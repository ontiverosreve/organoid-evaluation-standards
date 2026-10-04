"""Redo C3 cross-day transfer with filename-based pattern alignment.

The main run aligned patterns by patid index, which misaligns when the paired
file set differs between days (e.g. c7: day1 has {10X,3L,5X,8L}, day3 has
{10X,8L} -> patid 1 = 3L on day1 but 8L on day3). This script aligns by SSD
filename instead.
"""
import os, re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT = os.path.expanduser('~/workspace/reservoir-eeg/work/audit_shao2025')
CACHE = os.path.join(OUT, 'cache')
RES = os.path.join(OUT, 'results')

def norm_stem(name):
    s = name.lower()
    if s.endswith('.mat'): s = s[:-4]
    return re.sub(r'[_-]event$', '', s)

def file_key(name):
    s = norm_stem(name)
    mnum = re.search(r'(\d+)', s)
    mlet = re.search(r'([lxcsy])\d*$', s)
    return (int(mnum.group(1)) if mnum else -1,
            mlet.group(1).upper() if mlet else '?')

pairs = pd.read_csv(os.path.join(RES, 'file_pairs.csv'))
pairs = pairs[pairs.event_file.notna()].copy()
pairs = pairs.sort_values(['experiment', 'culture', 'day', 'ssd_file'])

def pat_files(exp, cult, day):
    sub = pairs[(pairs.experiment == exp) & (pairs.culture == cult) & (pairs.day == day)]
    return list(sub.ssd_file)

def pat_keys(exp, cult, day):
    return [file_key(f) for f in pat_files(exp, cult, day)]

def load(exp, cult, day):
    p = os.path.join(CACHE, f'{exp}_c{cult}_d{day}.npz')
    if not os.path.exists(p): return None
    return dict(np.load(p, allow_pickle=True))

def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))

rows = []
for exp in ['two_pattern', 'six_pattern']:
    for cult in range(1, 11):
        d1, d3 = load(exp, cult, 1), load(exp, cult, 3)
        if d1 is None or d3 is None: continue
        f1, f3 = pat_files(exp, cult, 1), pat_files(exp, cult, 3)
        k1, k3 = pat_keys(exp, cult, 1), pat_keys(exp, cult, 3)
        # align by (pattern-number, family-letter) key; fall back to filename
        common_idx = []
        used3 = set()
        for a, ka in enumerate(k1):
            for b, kb in enumerate(k3):
                if b in used3: continue
                if ka == kb or f1[a] == f3[b]:
                    common_idx.append((a, b)); used3.add(b); break
        if len(common_idx) < 2: continue
        i1 = [a for a, b in common_idx]
        i3 = [b for a, b in common_idx]
        keynames = '|'.join(f'{k1[a][0]}{k1[a][1]}' for a, b in common_idx)
        u1 = {tuple(u): i for i, u in enumerate(np.asarray(d1['units']))}
        u3 = {tuple(u): i for i, u in enumerate(np.asarray(d3['units']))}
        shared = sorted(set(u1) & set(u3))
        if len(shared) < 5: continue
        j1 = [u1[u] for u in shared]; j3 = [u3[u] for u in shared]
        X1 = np.asarray(d1['X_paper'], dtype=np.float64)[:, j1]
        X3 = np.asarray(d3['X_paper'], dtype=np.float64)[:, j3]
        y1 = np.asarray(d1['patid']); y3 = np.asarray(d3['patid'])
        # select trials of common patterns, remap to 0..k-1 by common order
        def sel(X, y, idx):
            keep = np.isin(y, idx)
            remap = {old: new for new, old in enumerate(idx)}
            return X[keep], np.array([remap[v] for v in y[keep]])
        X1s, y1s = sel(X1, y1, i1)
        X3s, y3s = sel(X3, y3, i3)
        m = clf().fit(X1s, y1s); a13 = m.score(X3s, y3s)
        m = clf().fit(X3s, y3s); a31 = m.score(X1s, y1s)
        rows.append(dict(experiment=exp, culture=cult, n_shared_units=len(shared),
            n_patterns=len(common_idx), patterns=keynames,
            train_d1_test_d3=round(a13, 4), train_d3_test_d1=round(a31, 4),
            chance=round(1/len(common_idx), 4)))
        print(f'{exp} c{cult}: k={len(common_idx)} d1->d3={a13:.3f} d3->d1={a31:.3f}', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, 'cross_day_transfer_v2.csv'), index=False)
for exp in ['two_pattern', 'six_pattern']:
    s = df[df.experiment == exp]
    print(exp, 'd1->d3 mean %.3f | d3->d1 mean %.3f' % (s.train_d1_test_d3.mean(), s.train_d3_test_d1.mean()))
