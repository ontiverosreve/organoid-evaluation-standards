"""Audit #4 (Shao et al. 2025): build per-trial features and cache them.

For each (experiment, culture, day): pairs each SSD spike file with its EVENT
stim-time file, then for every stimulation trial counts spikes per sorted unit
in several post-stimulus windows. Caches int16 matrices to npz.

Windows replicate the paper's own feature code (EvokedResponseMatrix.py):
    valid_count = sum((spikes >= stim+0.01) & (spikes <= stim+0.05))
i.e. the paper ALREADY blanks the first 10 ms post-stimulus.
"""
import os, re, sys
import numpy as np
import scipy.io as sio

SRC = os.path.expanduser('~/workspace/reservoir-eeg/work/dataset-sweep/shao2025/pattern_recognition')
OUT = os.path.expanduser('~/workspace/reservoir-eeg/work/audit_shao2025')
CACHE = os.path.join(OUT, 'cache')
os.makedirs(CACHE, exist_ok=True)

WINDOWS = {
    'paper':   (0.010, 0.050),   # paper's own window (10-50 ms)
    'noblank': (0.000, 0.050),   # no artifact blanking
    'artifact':(0.000, 0.002),   # artifact-only control
    'early':   (0.002, 0.050),   # 2 ms blank
    'wide':    (0.010, 0.200),
    'vwide':   (0.010, 0.500),
}

def norm_stem(name):
    s = name.lower()
    if s.endswith('.mat'): s = s[:-4]
    s = re.sub(r'[_-]event$', '', s)
    return s

def file_key(name):
    """(first_number, family_letter) for matching SSD<->EVENT.
    The family letter is the letter at the end of the stem (optionally
    followed by digits), e.g. 10X->X, 2_first_learning_L1->L,
    spikeSortingData10L->L, 10-X2->X."""
    s = norm_stem(name)
    mnum = re.search(r'(\d+)', s)
    mlet = re.search(r'([lxcsy])\d*$', s)
    num = int(mnum.group(1)) if mnum else -1
    let = mlet.group(1).upper() if mlet else '?'
    return num, let

def family_letter(name):
    s = norm_stem(name)
    m = re.search(r'([lxcsy])\d*$', s)
    return m.group(1).upper() if m else '?'

def load_spikes(path):
    d = sio.loadmat(path, simplify_cells=True)
    units, unsorted = {}, {}
    for k, v in d.items():
        if k.startswith('__'): continue
        arr = np.ravel(np.asarray(v, dtype=np.float64))
        m1 = re.search(r'ID_(\d+)', k); m2 = re.search(r'cluster(\d+)', k)
        if 'unsorted' in k:
            eid = int(m1.group(1)) if m1 else -1
            unsorted.setdefault(eid, []).append(arr)
        elif m1 and m2:
            units[(int(m1.group(1)), int(m2.group(1)))] = arr
    return units, unsorted

def main():
    pair_log = []
    for exp in ['two_pattern', 'six_pattern']:
        for cult in range(1, 11):
            for day in range(1, 4):
                ssd_dir = os.path.join(SRC, exp, f'culture_{cult}', f'day_{day}', 'SSD')
                ev_dir  = os.path.join(SRC, exp, f'culture_{cult}', f'day_{day}', 'EVENT')
                if not os.path.isdir(ssd_dir): continue
                ssd_files = sorted(f for f in os.listdir(ssd_dir) if f.endswith('.mat'))
                ev_files  = sorted(f for f in os.listdir(ev_dir) if f.endswith('.mat')) if os.path.isdir(ev_dir) else []
                # Two-pass matching: (1) exact normalized-stem equality, each EVENT
                # file used at most once; (2) key-based fallback for leftovers.
                ssd_stems = {f: norm_stem(f) for f in ssd_files}
                ev_stems = {f: norm_stem(f) for f in ev_files}
                ev_by_stem = {}
                for f, s in ev_stems.items():
                    ev_by_stem.setdefault(s, []).append(f)
                ev_by_key = {}
                for f in ev_files:
                    ev_by_key.setdefault(file_key(f), []).append(f)
                used_ev = set()
                pairs = []
                # pass 1: exact stem match
                for sf in ssd_files:
                    s = ssd_stems[sf]
                    cands = [f for f in ev_by_stem.get(s, []) if f not in used_ev]
                    if len(cands) == 1:
                        pairs.append((sf, cands[0], ''))
                        used_ev.add(cands[0])
                        pair_log.append((exp, cult, day, sf, cands[0], ''))
                    elif cands:
                        pair_log.append((exp, cult, day, sf, None, 'AMBIGUOUS-exact'))
                    else:
                        pair_log.append((exp, cult, day, sf, None, 'no-exact-match'))
                # pass 2: key-based for unmatched SSD files
                paired_ssd = set(sf for sf, _, _ in pairs)
                for sf in ssd_files:
                    if sf in paired_ssd: continue
                    k = file_key(sf)
                    cands = [f for f in ev_by_key.get(k, []) if f not in used_ev]
                    flag = ''
                    if not cands:
                        cands = [f for kk, fl in ev_by_key.items() if kk[0] == k[0]
                                 for f in fl if f not in used_ev]
                        if cands: flag = 'number-only-match'
                    if len(cands) == 1:
                        pairs.append((sf, cands[0], flag))
                        used_ev.add(cands[0])
                        # fix the log row for this file
                        for i in range(len(pair_log) - 1, -1, -1):
                            if (pair_log[i][0] == exp and pair_log[i][1] == cult
                                    and pair_log[i][2] == day and pair_log[i][3] == sf):
                                pair_log[i] = (exp, cult, day, sf, cands[0], flag)
                                break
                    elif not cands:
                        pass  # already logged as no-exact-match
                    else:
                        for i in range(len(pair_log) - 1, -1, -1):
                            if (pair_log[i][0] == exp and pair_log[i][1] == cult
                                    and pair_log[i][2] == day and pair_log[i][3] == sf):
                                pair_log[i] = (exp, cult, day, sf, None, 'AMBIGUOUS-key')
                                break
                if not pairs:
                    continue
                # union of unit keys across paired patterns
                all_units = {}
                pat_data = []
                for sf, ef, flag in pairs:
                    units, _ = load_spikes(os.path.join(ssd_dir, sf))
                    ed = sio.loadmat(os.path.join(ev_dir, ef), simplify_cells=True)
                    ek = [k for k in ed if not k.startswith('__')][0]
                    stims = np.ravel(np.asarray(ed[ek], dtype=np.float64))
                    pat_data.append((sf, family_letter(sf), units, stims, flag))
                    for u in units: all_units[u] = True
                ukeys = sorted(all_units)
                uidx = {u: i for i, u in enumerate(ukeys)}
                n_units = len(ukeys)
                Xs, ys, fams, nstims = {}, {}, {}, {}
                for wname, (a, b) in WINDOWS.items():
                    Xs[wname] = []
                all_fams, all_patids = [], []
                for pi, (sf, fam, units, stims, flag) in enumerate(pat_data):
                    n = len(stims)
                    M = {w: np.zeros((n, n_units), dtype=np.int16) for w in WINDOWS}
                    for u, arr in units.items():
                        j = uidx[u]
                        # searchsorted per window
                        for wname, (a, b) in WINDOWS.items():
                            lo = np.searchsorted(arr, stims + a, side='left')
                            hi = np.searchsorted(arr, stims + b, side='right')
                            M[wname][:, j] = (hi - lo).astype(np.int16)
                    for wname in WINDOWS:
                        Xs[wname].append(M[wname])
                    all_fams += [fam] * n
                    all_patids += [pi] * n
                    nstims[sf] = n
                out = dict(units=np.array(ukeys, dtype=np.int64),
                           fam=np.array(all_fams), patid=np.array(all_patids, dtype=np.int64))
                for wname in WINDOWS:
                    out['X_' + wname] = np.vstack(Xs[wname]).astype(np.int16)
                np.savez_compressed(os.path.join(CACHE, f'{exp}_c{cult}_d{day}.npz'), **out)
                print(f'{exp} c{cult} d{day}: {len(pairs)} pairs, {out["X_paper"].shape[0]} trials, '
                      f'{n_units} units, fams={sorted(set(all_fams))}', flush=True)
    import csv
    with open(os.path.join(OUT, 'results', 'file_pairs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'culture', 'day', 'ssd_file', 'event_file', 'flag'])
        w.writerows(pair_log)
    print('pair log rows:', len(pair_log))

if __name__ == '__main__':
    main()
