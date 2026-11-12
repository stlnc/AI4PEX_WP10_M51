"""
Scatterplot: heavy-rainfall threshold (q50–q95) vs % of those days inside ExEvEs.
Grasslands in Europe, 3 ET scenarios.
Output: exeves_europe_prec_quantile_scatter_grasslands.png
"""
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pyreadr
import warnings; warnings.filterwarnings('ignore')

GLEAM_FILE   = "gleam_e_mm_land_198001_202412_025_daily.nc"
FLUXCOM_FILE = "ET_2001_2020_025_daily_eur.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_land_197901_202012_025_daily.nc"
MASKS_FILE   = "masks.rda"

SC_KEYS   = ['S1', 'S2', 'S3']
SC_LABELS = ["GLEAM 1980–2020", "GLEAM 2001–2020", "FLUXCOM 2001–2020"]
SC_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
SEASON_MONTHS = [4, 5, 6, 7, 8, 9, 10]
QUANTILES = np.arange(50, 96, 5)  # 50, 55, 60, ..., 95


# ── Detection ─────────────────────────────────────────────────────────────────

def pentad_zscore(et, times_pd):
    T, N = et.shape
    pentad = np.minimum((times_pd.dayofyear.values - 1) // 5, 72)
    z = np.full_like(et, np.nan, dtype=np.float32)
    for p in range(73):
        idx = pentad == p
        if not idx.any(): continue
        v = et[idx]
        mu = np.nanmean(v, axis=0); sg = np.nanstd(v, axis=0)
        z[idx] = (v - mu) / np.where(sg > 0, sg, np.nan)
    return z


def detect_exeves(et, times_pd):
    T, N = et.shape
    z    = pentad_zscore(et, times_pd)
    z80  = np.nanpercentile(z, 80, axis=0)
    z95  = np.nanpercentile(z, 95, axis=0)
    above = z >= z80; extreme = z >= z95
    del z
    out = np.zeros((T, N), dtype=bool)
    for j in range(N):
        if not above[:, j].any(): continue
        d  = np.diff(above[:, j].astype(np.int8), prepend=0, append=0)
        ss = np.where(d ==  1)[0]; es = np.where(d == -1)[0]
        for s, e in zip(ss, es):
            if (e - s) >= 2 and extreme[s:e, j].any():
                out[s:e, j] = True
    return out


# ── Load masks — grasslands only ──────────────────────────────────────────────

print("Loading masks ...")
masks = pyreadr.read_r(MASKS_FILE)['masks']
eur = masks[
    masks['ipcc_continent'].isin(['EUROPE', 'EUROPE-AFRICA']) &
    (masks['land_mask'] == 'land') &
    (masks['land_cover_short_class'] == 'Grasslands')
].copy().reset_index(drop=True)
del masks
print(f"  Grassland cells: {len(eur)}")


# ── Load datasets ─────────────────────────────────────────────────────────────

def select_period_apr_oct(da, t0, t1):
    sub = da.sel(time=slice(t0, t1))
    return sub.where(sub.time.dt.month.isin(SEASON_MONTHS), drop=True)

lat_min = eur.lat.min() - 0.5;  lat_max = eur.lat.max() + 0.5
lon_min = eur.lon.min() - 0.5;  lon_max = eur.lon.max() + 0.5

print("Loading GLEAM ...")
gl_ds    = xr.open_dataset(GLEAM_FILE)
gl_ref   = gl_ds['E'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
eur_lats = gl_ref.lat.values
eur_lons = gl_ref.lon.values
gl_s1_da = select_period_apr_oct(gl_ref, '1980-01-01', '2020-12-31')
gl_s1    = gl_s1_da.values.astype(np.float32)
t_s1     = pd.DatetimeIndex(gl_s1_da.time.values)
gl_ds.close()

print("Loading MSWEP ...")
ms_ds   = xr.open_dataset(MSWEP_FILE)
ms_eur  = ms_ds['mask'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
ms_full = ms_eur.where(ms_eur.time.dt.month.isin(SEASON_MONTHS), drop=True).values.astype(np.float32)
ms_full[ms_full <= 0] = np.nan  # wet-day-only basis

print("Computing quantile thresholds (q50–q95) ...")
q_grids = {int(q): np.nanpercentile(ms_full, q, axis=0) for q in QUANTILES}
q_valid = np.isfinite(q_grids[95])
del ms_full

ms_s1_da = select_period_apr_oct(ms_eur, '1980-01-01', '2020-12-31')
ms_s1    = ms_s1_da.values.astype(np.float32)
ms_ds.close()

t_s2_start = t_s1.searchsorted(pd.Timestamp('2001-01-01'))
gl_s2 = gl_s1[t_s2_start:]
ms_s2 = ms_s1[t_s2_start:]
t_s2  = t_s1[t_s2_start:]

print("Loading FLUXCOM ...")
fc_ds    = xr.open_dataset(FLUXCOM_FILE)
fc_land  = fc_ds["land_fraction"].values >= 0.5
fc_s3_da = select_period_apr_oct(fc_ds['ET'], '2001-01-01', '2020-12-31')
t_s3     = pd.DatetimeIndex(fc_s3_da.time.values)
fc_s3    = (fc_s3_da.values * 24).astype(np.float32)
fc_ds.close()

print(f"S1: {gl_s1.shape}  S2: {gl_s2.shape}  S3: {fc_s3.shape} (Apr–Oct only)")


# ── Grid index mapping ────────────────────────────────────────────────────────

dlat = eur_lats[1] - eur_lats[0]
dlon = eur_lons[1] - eur_lons[0]
nlat, nlon = len(eur_lats), len(eur_lons)

i_lat = np.round((eur.lat.values - eur_lats[0]) / dlat).astype(int)
i_lon = np.round((eur.lon.values - eur_lons[0]) / dlon).astype(int)

ok    = (i_lat >= 0) & (i_lat < nlat) & (i_lon >= 0) & (i_lon < nlon)
eur   = eur[ok].copy().reset_index(drop=True)
i_lat = i_lat[ok]; i_lon = i_lon[ok]
q_valid_c = q_valid[i_lat, i_lon]
print(f"Grassland cells in bounds: {len(eur)}")

# Quantile threshold vectors for each grassland cell
q_thresh_c = {q: q_grids[q][i_lat, i_lon] for q in QUANTILES}


# ── Compute % heavy-rainfall days in ExEvEs per cell × quantile ───────────────

records = []  # list of (scenario_label, quantile, pct_per_cell)

for sc_label, sc_key, et_grid, times_pd, ms_grid in [
    (SC_LABELS[0], 'S1', gl_s1, t_s1, ms_s1),
    (SC_LABELS[1], 'S2', gl_s2, t_s2, ms_s2),
    (SC_LABELS[2], 'S3', fc_s3, t_s3, ms_s2),
]:
    print(f"\n{sc_key}: {et_grid.shape[0]} days ...")
    et_cells = et_grid[:, i_lat, i_lon]   # (T, N)
    ms_cells = ms_grid[:, i_lat, i_lon]   # (T, N)

    exeve = detect_exeves(et_cells, times_pd)  # (T, N) bool

    has_data = ~np.all(np.isnan(et_cells), axis=0)
    if sc_key == 'S3':
        valid = has_data & q_valid_c & fc_land[i_lat, i_lon]
    else:
        valid = has_data & q_valid_c

    for q in QUANTILES:
        thresh = q_thresh_c[q][np.newaxis, :]         # (1, N)
        heavy  = ms_cells >= thresh                    # (T, N) bool
        n_h    = heavy.sum(axis=0).astype(np.float32) # (N,)
        n_hw   = (exeve & heavy).sum(axis=0).astype(np.float32)
        ok_q   = valid & (n_h > 0)
        pct    = np.where(ok_q, n_hw / n_h * 100, np.nan)

        valid_pcts = pct[ok_q]
        n_cells    = valid_pcts.size
        records.append({
            'scenario': sc_label,
            'quantile': int(q),
            'pct_all': valid_pcts,       # array — one value per valid cell
            'mean': float(np.nanmean(valid_pcts)),
            'p25':  float(np.nanpercentile(valid_pcts, 25)),
            'p75':  float(np.nanpercentile(valid_pcts, 75)),
            'n':    n_cells,
        })
        print(f"  q{q:02d}  mean={np.nanmean(valid_pcts):.1f}%  n={n_cells}")

    del et_cells, ms_cells, exeve

del gl_s1, ms_s1, fc_s3

df_stats = pd.DataFrame([{k: v for k, v in r.items() if k != 'pct_all'}
                          for r in records])


# ── Plot ──────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(8, 7))

for sc_label, color in zip(SC_LABELS, SC_COLORS):
    sc_recs = [r for r in records if r['scenario'] == sc_label]

    # Raw cell scatter (jitter y for visibility)
    for r in sc_recs:
        n = r['pct_all'].size
        jitter = rng.uniform(-1.8, 1.8, n)
        ax.scatter(r['pct_all'], np.full(n, r['quantile']) + jitter,
                   color=color, alpha=0.08, s=4, linewidths=0, rasterized=True)

    # Mean ± IQR line
    sub = df_stats[df_stats['scenario'] == sc_label].sort_values('quantile')
    ax.plot(sub['mean'], sub['quantile'],
            color=color, lw=2.5, marker='o', markersize=6, zorder=5, label=sc_label)
    ax.fill_betweenx(sub['quantile'], sub['p25'], sub['p75'],
                     color=color, alpha=0.15)

ax.set_ylabel("Precipitation quantile threshold", fontsize=12)
ax.set_xlabel("% of heavy-rainfall days falling within ExEvEs", fontsize=12)
ax.set_yticks(QUANTILES)
ax.set_yticklabels([f"q{q}" for q in QUANTILES], fontsize=10)
ax.xaxis.set_tick_params(labelsize=10)
ax.legend(fontsize=10, framealpha=0.8)
ax.set_title("Heavy-rainfall days during ExEvEs\nGrasslands, Europe (Apr–Oct)",
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
fig.tight_layout()
fig.savefig("exeves_europe_prec_quantile_scatter_grasslands.png",
            dpi=150, bbox_inches='tight')
print("\nSaved exeves_europe_prec_quantile_scatter_grasslands.png")
