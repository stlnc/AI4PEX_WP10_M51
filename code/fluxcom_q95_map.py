import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

FLUXCOM_FILE = "ET_2001_2021_025_daily_med.nc"
MSWEP_FILE   = "mswep-v2-8_tp_mm_med_197901_202012_025_daily.nc"


def pentad_zscore(et, times_pd):
    T, nlat, nlon = et.shape
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
    T, nlat, nlon = et.shape; N = nlat * nlon
    z   = pentad_zscore(et, times_pd).reshape(T, N)
    z80 = np.nanpercentile(z, 80, axis=0)
    z95 = np.nanpercentile(z, 95, axis=0)
    above = z >= z80; extreme = z >= z95
    out = np.zeros((T, N), dtype=bool)
    for j in range(N):
        if not above[:, j].any(): continue
        d  = np.diff(above[:, j].astype(np.int8), prepend=0, append=0)
        ss = np.where(d ==  1)[0]; es = np.where(d == -1)[0]
        for s, e in zip(ss, es):
            if (e - s) >= 2 and extreme[s:e, j].any():
                out[s:e, j] = True
    return out.reshape(T, nlat, nlon)


print("Loading ...")
fc_ds    = xr.open_dataset(FLUXCOM_FILE)
land_2d  = fc_ds["land_fraction"].values >= 0.5
ref_lat  = fc_ds.lat.values
ref_lon  = fc_ds.lon.values
fc_et    = (fc_ds["ET"].values * 24).astype(np.float32)
fc_times = pd.DatetimeIndex(fc_ds.time.values)

ms_ds    = xr.open_dataset(MSWEP_FILE)
ms_np    = ms_ds["mask"].values.astype(np.float32)
ms_times = pd.DatetimeIndex(ms_ds.time.values)

t0, t1 = "2001-01-01", "2020-12-31"
fc_idx = np.where((fc_times >= pd.Timestamp(t0)) & (fc_times <= pd.Timestamp(t1)))[0]
ms_idx = np.where((ms_times >= pd.Timestamp(t0)) & (ms_times <= pd.Timestamp(t1)))[0]

fc_sub = fc_et[fc_idx]
ms_sub = ms_np[ms_idx]
t_pd   = pd.DatetimeIndex(fc_times[fc_idx])

print("Detecting FLUXCOM ExEvEs ...")
exeve = detect_exeves(fc_sub, t_pd)

q95_vals = np.nanpercentile(ms_np, 95, axis=0)
q95_valid = q95_vals > 0.1
q95m  = ms_sub >= q95_vals[np.newaxis]

fc_has_data = ~np.all(np.isnan(fc_sub), axis=0)
valid = q95_valid & land_2d & fc_has_data

n_q  = q95m.sum(axis=0).astype(float)
n_qw = (exeve & q95m).sum(axis=0).astype(float)
pct_q95 = np.where((n_q > 0) & valid, n_qw / n_q * 100, np.nan)

from scipy.stats import gaussian_kde

mu = np.nanmean(pct_q95)
print(f"Mean % q95 during FLUXCOM ExEvEs: {mu:.1f}%")

lon2d, lat2d = np.meshgrid(ref_lon, ref_lat)
vmax = np.nanpercentile(pct_q95[valid], 97)
v = pct_q95[np.isfinite(pct_q95)].ravel()

fig, (ax_map, ax_den) = plt.subplots(1, 2, figsize=(15, 4))

im = ax_map.pcolormesh(lon2d, lat2d, pct_q95, cmap="YlOrRd", vmin=0, vmax=vmax)
plt.colorbar(im, ax=ax_map, label="%")
ax_map.set_title("% of q95 days during FLUXCOM ExEvEs (2001–2020)", fontsize=11)
ax_map.set_xlabel("Longitude"); ax_map.set_ylabel("Latitude")

ax_den.hist(v, bins=60, range=(0, 100), density=True, alpha=0.35, color="darkorange")
kde = gaussian_kde(v, bw_method=0.15)
x = np.linspace(0, 100, 500)
ax_den.plot(x, kde(x), color="darkorange", lw=2, label=f"FLUXCOM  mean={mu:.1f}%")
ax_den.axvline(mu, color="darkorange", lw=1.2, ls="--")
ax_den.set_xlabel("Percentage (%)")
ax_den.set_ylabel("Density")
ax_den.set_title("% of q95 days during FLUXCOM ExEvEs (per land cell)", fontsize=11)
ax_den.legend(fontsize=9)
ax_den.set_xlim(0, 100)
ax_den.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("fluxcom_q95_map.png", dpi=150, bbox_inches="tight")
print("Saved fluxcom_q95_map.png")
