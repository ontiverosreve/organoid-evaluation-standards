"""Audit #6: reimplementation of Liu & Buonomano Fig-2 event-detection pipeline.

Faithful port of Figure2_timing_detection_catAllTraces_V5_.m +
SMOOTHTRACE.m + CenterOfGravity.m. Conventions:
- dt = 0.1 ms per sample; traces are (n_traces, 30000)
- analysis window: samples 0..20000 (0-2000 ms); stimulus at sample 10000 (1000 ms)
- event times reported in ms relative to window start (stimulus), matching the paper
"""
import numpy as np


def smoothtrace(trace, window):
    """Moving average with +/- window points, zero-padded; first/last `window`
    points replaced by the original (matches SMOOTHTRACE.m)."""
    trace = np.asarray(trace, dtype=float)
    n = len(trace)
    padded = np.zeros(n + 2 * window)
    padded[window:window + n] = trace
    dummy = np.zeros(n)
    for i in range(-window, window + 1):
        dummy += padded[window + i: window + i + n]
    dummy /= (2 * window + 1)
    dummy[:window] = trace[:window]
    dummy[n - window:] = trace[n - window:]
    return dummy


def center_of_gravity(data):
    """Port of CenterOfGravity.m. data: (n_traces, tmax). Returns per-trace
    index (0-based) where cumulative sum crosses half the integral, after
    shifting each trace by |min| (as in the .m code)."""
    data = np.asarray(data, dtype=float)
    traces = data + np.abs(data.min(axis=1, keepdims=True))
    integral = traces.sum(axis=1, keepdims=True)
    mid = integral / 2.0
    cumul = np.cumsum(traces, axis=1)
    abovemid = (cumul - mid) > 0
    d = np.diff(abovemid.astype(int), axis=1)
    # first index where diff == 1
    centers = np.argmax(d == 1, axis=1)
    # argmax returns 0 when never true; guard: check any
    has = (d == 1).any(axis=1)
    centers = np.where(has, centers, -1)
    return centers  # 0-based index into data


def detect_events(keep_trace, dt=0.1, std_threshold=1.0, smooth_ms=10.0,
                  shift_ms=-10.0, baseline_end_ms=1000.0, window_end_ms=2000.0):
    """Port of the slope-detection block. Returns dict with per-trace first
    event indices (ms, relative to window start) and summary stats."""
    n_traces, n_samp = keep_trace.shape
    w = int(round(smooth_ms / dt))
    shift = int(round(shift_ms / dt))
    w0, w1 = 0, int(round(window_end_ms / dt))
    b0 = int(round(baseline_end_ms / dt))  # 10000
    keeptraces = keep_trace[:, w0:w1]
    # concatenate all traces, smooth, shift, diff
    tracesignal = keeptraces.reshape(-1)
    smooth = smoothtrace(tracesignal, w)
    shifted = np.roll(smooth, shift)
    shiftdiff = shifted - smooth
    detected = shiftdiff > 3.0 * np.std(shiftdiff)
    tracelen = keeptraces.shape[1]
    detectedmat = detected.reshape(n_traces, tracelen)
    sub = detectedmat[:, b0:w1]  # analysis subwindow
    all_events = []
    for t in range(n_traces):
        idx = np.flatnonzero(sub[t])
        if len(idx) == 0:
            continue
        # first index of each run of consecutive detections
        firsts = idx[np.concatenate([[True], np.diff(idx) > 1])]
        all_events.extend(firsts.tolist())
    all_events = np.array(all_events, dtype=float)
    median_event_ms = float(np.median(all_events)) * dt if len(all_events) else np.nan
    # peak time / amplitude on smoothed per-trace window
    peak_idx, peak_amp = [], []
    for t in range(n_traces):
        s = smoothtrace(keeptraces[t, b0:w1], w)
        i = int(np.argmax(s))
        peak_idx.append(i); peak_amp.append(float(s[i]))
    baseline = float(np.mean(keep_trace[:, :b0]))
    return {
        'median_event_ms': median_event_ms,
        'n_events': len(all_events),
        'peak_time_ms': float(np.median(peak_idx)) * dt,
        'peak_amp': float(np.median(peak_amp)) - baseline,
        'baseline': baseline,
    }


def cog_ms(keep_trace, dt=0.1, baseline_end_ms=1000.0, window_end_ms=1600.0):
    """Center of gravity over [1000,1600] ms per trace; returns mean ms
    relative to window start (matches compMETA.CenterOfGravity — the stored
    values were computed on the 600 ms window noted in the .m comments, not
    the 2000 ms cogwindowEnd in the script)."""
    b0 = int(round(baseline_end_ms / dt)); w1 = int(round(window_end_ms / dt))
    centers = center_of_gravity(keep_trace[:, b0:w1])
    valid = centers[centers >= 0]
    return float(np.mean(valid)) * dt if len(valid) else np.nan
