"""Audit #6 plots."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(WORK, 'results'); PLT = os.path.join(WORK, 'plots')
rows = list(csv.DictReader(open(os.path.join(RES, 'features.csv'))))
for r in rows:
    r['event_ms'] = float(r['event_ms'])
batches = sorted(set(r['date'] for r in rows))
cmap = plt.cm.tab10
bcol = {b: cmap(i) for i, b in enumerate(batches)}

# 1. event time by protocol, colored by batch
fig, ax = plt.subplots(figsize=(9, 5))
for j, proto in enumerate(['12EARLY', 'VERYLATE']):
    sel = [r for r in rows if r['protocol'] == proto]
    xs = [j + (hash(r['date']) % 100) / 100 * 0.6 - 0.3 for r in sel]
    ax.scatter(xs, [r['event_ms'] for r in sel],
               c=[bcol[r['date']] for r in sel], s=60, alpha=0.85,
               edgecolors='k', linewidths=0.5, label=None)
    med = np.median([r['event_ms'] for r in sel])
    ax.hlines(med, j - 0.35, j + 0.35, colors='k', lw=2.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Early (10 ms ISI)', 'Late (370 ms ISI)'])
ax.set_ylabel('Median event time (ms, post-stimulus)')
ax.set_title('Evoked event time by training protocol — colored by batch (n=65 cells, 7 batches)')
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=bcol[b],
                       markeredgecolor='k', markersize=8, label=b) for b in batches]
ax.legend(handles=handles, title='batch (YYMMDD)', ncol=4, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PLT, 'event_time_by_batch.png'), dpi=120)

# 2. cross-batch transfer
t = list(csv.DictReader(open(os.path.join(RES, 'cross_batch_transfer.csv'))))
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(len(t))
accs = [float(r['acc']) for r in t]
cols = ['tab:green' if a >= 0.8 else 'tab:orange' for a in accs]
ax.bar(xs, accs, color=cols, edgecolor='k')
ax.axhline(0.5, color='k', ls='--', label='chance')
ax.axhline(np.mean(accs), color='tab:red', ls='-', label=f"mean {np.mean(accs):.2f}")
ax.set_xticks(xs); ax.set_xticklabels([r['batch'] for r in t], rotation=30)
ax.set_ylabel('Protocol-decode accuracy (held-out batch)')
ax.set_title('Leave-one-batch-out transfer (§3.7): protocol decodes across batches')
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(os.path.join(PLT, 'cross_batch_transfer.png'), dpi=120)

# 3. feature-set comparison
f = list(csv.DictReader(open(os.path.join(RES, 'rate_baseline.csv'))))
fig, ax = plt.subplots(figsize=(7, 4))
names = [r['feature_set'] for r in f]; ms = [float(r['mean_acc']) for r in f]
ss = [float(r['sd_acc']) for r in f]
ax.bar(names, ms, yerr=ss, capsize=5, color=['tab:blue', 'tab:orange', 'tab:green'], edgecolor='k')
ax.axhline(0.5, color='k', ls='--', label='chance')
ax.set_ylabel('CV accuracy'); ax.set_title('What carries the protocol signal? (§3.5)')
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(PLT, 'feature_sets.png'), dpi=120)
print("plots done")
