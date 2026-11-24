"""Shared two-dimensional ExEvE detection helpers."""

import numpy as np


def pentad_zscore(et, times_pd):
    T, N = et.shape
    pentad = np.minimum((times_pd.dayofyear.values - 1) // 5, 72)
    z = np.full_like(et, np.nan, dtype=np.float32)
    for p in range(73):
        idx = pentad == p
        if not idx.any():
            continue
        v = et[idx]
        mu = np.nanmean(v, axis=0)
        sg = np.nanstd(v, axis=0)
        z[idx] = (v - mu) / np.where(sg > 0, sg, np.nan)
    return z


def detect_exeves(et, times_pd):
    T, N = et.shape
    z = pentad_zscore(et, times_pd)
    z80 = np.nanpercentile(z, 80, axis=0)
    z95 = np.nanpercentile(z, 95, axis=0)
    above = z >= z80
    extreme = z >= z95
    del z
    out = np.zeros((T, N), dtype=bool)
    for j in range(N):
        if not above[:, j].any():
            continue
        d = np.diff(above[:, j].astype(np.int8), prepend=0, append=0)
        ss = np.where(d == 1)[0]
        es = np.where(d == -1)[0]
        for s, e in zip(ss, es):
            if (e - s) >= 2 and extreme[s:e, j].any():
                out[s:e, j] = True
    return out
