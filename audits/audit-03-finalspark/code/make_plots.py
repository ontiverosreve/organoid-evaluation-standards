"""Audit #3 summary plots (aggregates only, from results CSVs)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = os.path.expanduser('~/workspace/reservoir-eeg/work/audit_finalspark')
RES = os.path.join(W, 'results')
PLT = os.path.join(W, 'plots')
os.makedirs(PLT, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

# ---- Plot 1: cross-day transfer failure (both datasets) ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

# FS369 amplitude decoding
ax = axes[0]
labels = ['Pooled CV', 'Per-day CV\n(best)', 'Blanked\n[10,50]ms', 'Transfer\n02-20→02-28', 'Transfer\n02-28→02-20']
res_vals = [0.536, 0.856, 0.509, 0.322, 0.330]
lr_vals  = [0.725, 0.967, 0.700, 0.533, 0.367]
x = np.arange(len(labels))
ax.bar(x - 0.2, res_vals, 0.4, label='Reservoir (canonical)', color='#4472C4')
ax.bar(x + 0.2, lr_vals, 0.4, label='Logistic regression', color='#ED7D31')
ax.axhline(1/3, color='k', linestyle='--', linewidth=1, label='Chance (1/3)')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Accuracy')
ax.set_title('FS369: amplitude decoding (3-way)\nwithin-day works, cross-day fails')
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)

# FS437 protocol decoding / transfer
ax = axes[1]
labels2 = ['Protocol CV', 'Blanked\n[10,50]ms', 'Transfer fwd\nP2 recall', 'Transfer rev\nP2 recall*', 'Day-ID from\nbackground']
vals2 = [0.4348, 0.4019, 0.162, 1.000, 0.5106]
null2 = [0.3325, 0.3330, 0.421, 0.527, 0.333]
x = np.arange(len(labels2))
ax.bar(x - 0.2, vals2, 0.4, color='#4472C4', label='Observed')
ax.bar(x + 0.2, null2, 0.4, color='#A5A5A5', label='Shuffle null')
ax.axhline(1/3, color='k', linestyle='--', linewidth=1, label='Chance (1/3)')
ax.set_xticks(x); ax.set_xticklabels(labels2, fontsize=9)
ax.set_title('FS437: protocol decoding & transfer\n*reverse "1.00" is classifier bias (see report)')
ax.legend(fontsize=9)

fig.suptitle('Audit #3: cross-day transfer fails in both cultures (§3.7)', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLT, 'transfer_failure.svg'))
plt.close(fig)

# ---- Plot 2: evoked ratios vs honest null (FS437) ----
fig, ax = plt.subplots(figsize=(8, 4.5))
protos = ['P1 (3 µA)', 'P2 (1 µA)', 'P3 (5 µA)']
obs = [4.1667, 2.7316, 4.5592]
null_mean = [1.0028, 1.0081, 1.0003]
null_p95 = [1.0282, 1.0517, 1.0462]
x = np.arange(len(protos))
ax.bar(x - 0.2, obs, 0.4, color='#4472C4', label='Observed evoked ratio')
ax.bar(x + 0.2, null_mean, 0.4, color='#A5A5A5', label='Stim-shuffled null (mean)')
ax.errorbar(x + 0.2, null_mean, yerr=np.array(null_p95) - np.array(null_mean),
            fmt='none', color='k', capsize=4)
ax.axhline(1.0, color='k', linestyle='--', linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(protos)
ax.set_ylabel('Post-stim [0,50]ms / baseline rate')
ax.set_title('FS437: evoked responses survive the stim-shuffled null')
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(PLT, 'evoked_ratios.svg'))
plt.close(fig)

# ---- Plot 3: session identity decodability ----
fig, ax = plt.subplots(figsize=(8, 4.5))
labels3 = ['FS369 epoch\n(early/mid/late)', 'FS369 day-ID', 'FS437 day-ID\n(background)']
acc3 = [0.9867, 0.9215, 0.5106]
ch3 = [1/3, 1/22, 1/3]
x = np.arange(len(labels3))
ax.bar(x, acc3, 0.55, color='#C00000', label='Observed')
ax.bar(x, ch3, 0.55, color='#A5A5A5', alpha=0.6, label='Chance')
for i, (a, c) in enumerate(zip(acc3, ch3)):
    ax.text(i, a + 0.02, f'{a:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i, c + 0.02, f'{c:.3f}', ha='center', fontsize=9, color='#555555')
ax.set_xticks(x); ax.set_xticklabels(labels3)
ax.set_ylabel('Accuracy')
ax.set_title('Session identity is decodable from background activity alone')
ax.set_ylim(0, 1.12)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(PLT, 'session_identity.svg'))
plt.close(fig)

print('plots written:', sorted(os.listdir(PLT)))
