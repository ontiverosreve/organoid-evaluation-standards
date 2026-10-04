"""Audit #4 plots."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.expanduser('~/workspace/reservoir-eeg/work/audit_shao2025')
RES = os.path.join(OUT, 'results')
PLT = os.path.join(OUT, 'plots')
os.makedirs(PLT, exist_ok=True)

pat = pd.read_csv(os.path.join(RES, 'decoding_pattern.csv'))
tr = pd.read_csv(os.path.join(RES, 'training_claim.csv'))

# ---- Fig 1: day1 vs day3 accuracy per culture (paper window, strat5fold, N-way) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, exp, title in zip(axes,
                          ['two_pattern', 'six_pattern'],
                          ['two_pattern: 4-way pattern decoding', 'six_pattern: 6-way pattern decoding']):
    sub = pat[(pat.experiment == exp) & (pat.window == 'paper') &
              (pat.scheme == 'strat5fold')]
    # keep only culture-days with full pattern sets for the headline comparison
    full = {'two_pattern': 4, 'six_pattern': 6}[exp]
    piv = sub[sub.n_patterns == full].pivot(index='culture', columns='day', values='accuracy')
    piv = piv.dropna()
    xs = np.arange(len(piv))
    ax.plot(xs, piv[1].values, 'o-', label='day 1', ms=5)
    ax.plot(xs, piv[3].values, 's-', label='day 3', ms=5)
    for i in range(len(piv)):
        ax.plot([xs[i], xs[i]], [piv[1].values[i], piv[3].values[i]], 'k-', alpha=0.3)
    ax.set_xticks(xs); ax.set_xticklabels([f'c{c}' for c in piv.index], rotation=45, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel('accuracy')
    ax.set_title(title)
    ax.legend(fontsize=9)
    d = piv[3].values - piv[1].values
    ax.text(0.02, 0.98, f"n={len(piv)} paired cultures\nmean Δ(day3−day1)={d.mean():+.3f}",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
fig.suptitle('Pattern-decoding accuracy: day 1 vs day 3 (paper window [10,50] ms, 5-fold CV)')
fig.tight_layout()
fig.savefig(os.path.join(PLT, 'day1_vs_day3.png'), dpi=120)

# ---- Fig 2: window comparison (artifact control story) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
order = ['artifact', 'noblank', 'early', 'paper', 'wide', 'vwide']
labels = {'artifact': '[0,2]ms\nartifact-only', 'noblank': '[0,50]ms',
          'early': '[2,50]ms', 'paper': '[10,50]ms\n(paper)',
          'wide': '[10,200]ms', 'vwide': '[10,500]ms'}
for ax, exp, title in zip(axes, ['two_pattern', 'six_pattern'],
                          ['two_pattern (4-way)', 'six_pattern (6-way)']):
    sub = pat[(pat.experiment == exp) & (pat.scheme == 'strat5fold')]
    full = {'two_pattern': 4, 'six_pattern': 6}[exp]
    sub = sub[sub.n_patterns == full]
    vals = [sub[sub.window == w].accuracy.values for w in order]
    bp = ax.boxplot(vals, labels=[labels[w] for w in order], patch_artist=True)
    for b in bp['boxes']:
        b.set_facecolor('#aec7e8')
    ax.axhline(1/full, ls='--', c='k', alpha=0.5)
    ax.text(0.98, 0.06, f'chance={1/full:.2f}', transform=ax.transAxes, ha='right', fontsize=9)
    ax.set_ylabel('accuracy')
    ax.set_title(title)
fig.suptitle('Decoding accuracy vs post-stimulus window (5-fold CV, all culture-days)')
fig.tight_layout()
fig.savefig(os.path.join(PLT, 'window_comparison.png'), dpi=120)

# ---- Fig 3: cross-day transfer ----
xfer = pd.read_csv(os.path.join(RES, 'cross_day_transfer.csv'))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, exp, title in zip(axes, ['two_pattern', 'six_pattern'],
                          ['two_pattern', 'six_pattern']):
    sub = xfer[xfer.experiment == exp].sort_values('culture')
    if len(sub) == 0:
        ax.text(0.5, 0.5, 'no data', ha='center'); continue
    xs = np.arange(len(sub))
    ax.plot(xs, sub.train_d1_test_d3.values, 'o-', label='train d1 → test d3', ms=5)
    ax.plot(xs, sub.train_d3_test_d1.values, 's-', label='train d3 → test d1', ms=5)
    ax.axhline(sub.chance.values[0], ls='--', c='k', alpha=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([f'c{c}' for c in sub.culture], rotation=45, fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel('accuracy')
    ax.set_title(title); ax.legend(fontsize=9)
fig.suptitle('Cross-day transfer of pattern decoders (shared units only)')
fig.tight_layout()
fig.savefig(os.path.join(PLT, 'cross_day_transfer.png'), dpi=120)
print('plots saved')
