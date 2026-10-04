"""Replicate the paper repo's ACTUAL classifier (RBF-SVM, gamma='auto') with their
chronological block scheme, for comparison with the paper's reported numbers.
Paper text says logistic regression; repo code uses svm.SVC(kernel='rbf').
"""
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import stats

OUT = os.path.expanduser('~/workspace/reservoir-eeg/work/audit_shao2025')
CACHE = os.path.join(OUT, 'cache')
RES = os.path.join(OUT, 'results')

def load(exp, cult, day):
    p = os.path.join(CACHE, f'{exp}_c{cult}_d{day}.npz')
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None

rows = []
for exp, full in [('two_pattern', 4), ('six_pattern', 6)]:
    for cult in range(1, 11):
        for day in [1, 3]:
            d = load(exp, cult, day)
            if d is None: continue
            X = np.asarray(d['X_paper'], dtype=np.float64)
            y = np.asarray(d['patid'])
            if len(np.unique(y)) != full: continue
            accs = []
            for b in range(5):
                tr, te = [], []
                for p_ in np.unique(y):
                    idx = np.where(y == p_)[0]
                    blk = idx[b*80:(b+1)*80]
                    tr += blk[:60].tolist(); te += blk[60:].tolist()
                if len(te) == 0: continue
                m = make_pipeline(StandardScaler(),
                                  SVC(kernel='rbf', C=1.0, gamma='auto')).fit(X[tr], y[tr])
                accs.append(m.score(X[te], y[te]))
            rows.append(dict(experiment=exp, culture=cult, day=day,
                             svm_block_acc=round(float(np.mean(accs)), 4)))
            print(f'{exp} c{cult} d{day}: svm={np.mean(accs):.3f}', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, 'svm_replication.csv'), index=False)
for exp in ['two_pattern', 'six_pattern']:
    piv = df[df.experiment == exp].pivot(index='culture', columns='day', values='svm_block_acc').dropna()
    if 1 in piv.columns and 3 in piv.columns:
        d1, d3 = piv[1].values, piv[3].values
        t, p_ = stats.ttest_rel(d3, d1)
        print(f'{exp} SVM: n={len(piv)} day1={d1.mean():.4f} day3={d3.mean():.4f} '
              f'delta={np.mean(d3-d1):+.4f} t={t:.3f} p={p_:.4f}')
