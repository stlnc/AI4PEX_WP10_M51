import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

GLEAM_FILE = "gleam_e_mm_med_198001_202412_025_daily.nc"
ET_FILE = "ET_2001_2021_025_daily_med.nc"
OUT_FIG = "compare_gleam_fluxcom.png"
OUT_MAPS = "compare_gleam_fluxcom_maps.png"
OUT_TS = "compare_gleam_fluxcom_timeseries.png"

# ── Load and align ────────────────────────────────────────────────────────────
gleam = xr.open_dataset(GLEAM_FILE)
et = xr.open_dataset(ET_FILE)

# Subset GLEAM to ET period
gleam = gleam.sel(time=slice("2001-01-01", "2021-12-31"))

# Convert ET: mm hr-1 → mm day-1
et["ET"] = et["ET"] * 24
et["ET"].attrs["units"] = "mm day-1"

# Grids are nominally identical but lat/lon values differ by ~1 ULP.
# Select GLEAM by range then assign ET's exact coordinates so operations align.
gleam_aligned = gleam["E"].sel(
    lat=slice(float(et.lat.max()) + 1e-4, float(et.lat.min()) - 1e-4),
    lon=slice(float(et.lon.min()) - 1e-4, float(et.lon.max()) + 1e-4),
).assign_coords(lat=et.lat, lon=et.lon)
et_da = et["ET"]

# Land mask: keep only pixels where land_fraction >= 0.5
land_mask = et["land_fraction"] >= 0.5
gleam_aligned = gleam_aligned.where(land_mask)
et_da = et_da.where(land_mask)

print(f"Common shape: {et_da.shape}")  # (time, lat, lon)

# ── Temporal means ────────────────────────────────────────────────────────────
gleam_mean = gleam_aligned.mean("time")
et_mean = et_da.mean("time")
bias = et_mean - gleam_mean  # FLUXCOM minus GLEAM

# ── Monthly area-averaged time series ─────────────────────────────────────────
gleam_ts = gleam_aligned.mean(("lat", "lon")).resample(time="ME").mean()
et_ts = et_da.mean(("lat", "lon")).resample(time="ME").mean()

# ── Daily spatial means (for scatter) ────────────────────────────────────────
gleam_daily = gleam_aligned.mean(("lat", "lon"))
et_daily = et_da.mean(("lat", "lon"))

# ── Pixel-wise correlation ────────────────────────────────────────────────────
# Resample both to monthly before correlating (speed + noise reduction)
gleam_mo = gleam_aligned.resample(time="ME").mean()
et_mo = et_da.resample(time="ME").mean()

corr = xr.corr(gleam_mo, et_mo, dim="time")

# ── Annual cycle ──────────────────────────────────────────────────────────────
gleam_clim = gleam_aligned.groupby("time.month").mean()
et_clim = et_da.groupby("time.month").mean()
gleam_clim_ts = gleam_clim.mean(("lat", "lon"))
et_clim_ts = et_clim.mean(("lat", "lon"))

# ── Scatter stats on daily data ───────────────────────────────────────────────
g_vals = gleam_daily.values
e_vals = et_daily.values
mask = np.isfinite(g_vals) & np.isfinite(e_vals)
g_vals, e_vals = g_vals[mask], e_vals[mask]
slope, intercept, r, p, _ = stats.linregress(g_vals, e_vals)
rmse = np.sqrt(np.mean((e_vals - g_vals) ** 2))
bias_scalar = np.mean(e_vals - g_vals)

print(f"R={r:.3f}  RMSE={rmse:.3f} mm/day  Bias={bias_scalar:.3f} mm/day")

lon2d, lat2d = np.meshgrid(et.lon.values, et.lat.values)
cmap_et = "YlGn"

# ── Maps figure: 3 rows ───────────────────────────────────────────────────────
fig_maps, axes_maps = plt.subplots(3, 1, figsize=(10, 12))

vmax_et = max(float(gleam_mean.quantile(0.99)), float(et_mean.quantile(0.99)))

im = axes_maps[0].pcolormesh(lon2d, lat2d, gleam_mean.values, cmap=cmap_et, vmin=0, vmax=vmax_et)
plt.colorbar(im, ax=axes_maps[0], label="mm day⁻¹")
axes_maps[0].set_title("GLEAM mean ET (2001–2021)")
axes_maps[0].set_ylabel("Latitude")

im = axes_maps[1].pcolormesh(lon2d, lat2d, et_mean.values, cmap=cmap_et, vmin=0, vmax=vmax_et)
plt.colorbar(im, ax=axes_maps[1], label="mm day⁻¹")
axes_maps[1].set_title("FLUXCOM mean ET (2001–2021)")
axes_maps[1].set_ylabel("Latitude")

blim = np.nanpercentile(np.abs(bias.values[np.isfinite(bias.values)]), 95)
norm = TwoSlopeNorm(vmin=-blim, vcenter=0, vmax=blim)
im = axes_maps[2].pcolormesh(lon2d, lat2d, bias.values, cmap="RdBu_r", norm=norm)
plt.colorbar(im, ax=axes_maps[2], label="mm day⁻¹")
axes_maps[2].set_title("Bias: FLUXCOM − GLEAM (2001–2021)")
axes_maps[2].set_xlabel("Longitude"); axes_maps[2].set_ylabel("Latitude")

fig_maps.suptitle("GLEAM vs FLUXCOM — Mean ET & Bias — Mediterranean",
                  fontsize=13, fontweight="bold")
fig_maps.tight_layout()
fig_maps.savefig(OUT_MAPS, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_MAPS}")

# ── Time series figure: monthly mean + annual cycle ───────────────────────────
fig_ts, (ax_ts, ax_clim) = plt.subplots(2, 1, figsize=(12, 8))

ax_ts.plot(gleam_ts.time.values, gleam_ts.values, label="GLEAM", color="steelblue", lw=1.2)
ax_ts.plot(et_ts.time.values, et_ts.values, label="FLUXCOM", color="darkorange", lw=1.2, alpha=0.85)
ax_ts.set_title("Monthly mean ET (land pixels, 2001–2021)")
ax_ts.set_ylabel("mm day⁻¹")
ax_ts.set_xlabel("Date")
ax_ts.legend()

months = np.arange(1, 13)
ax_clim.plot(months, gleam_clim_ts.values, "o-", label="GLEAM", color="steelblue")
ax_clim.plot(months, et_clim_ts.values, "s-", label="FLUXCOM", color="darkorange")
ax_clim.set_xticks(months)
ax_clim.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax_clim.set_title("Annual cycle (land pixels, 2001–2021)")
ax_clim.set_ylabel("mm day⁻¹")
ax_clim.legend()

fig_ts.suptitle("GLEAM vs FLUXCOM — Mediterranean", fontsize=13, fontweight="bold")
fig_ts.tight_layout()
fig_ts.savefig(OUT_TS, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_TS}")

# ── Main figure: correlation map + daily scatter ──────────────────────────────
fig, (ax_corr, ax_scat) = plt.subplots(1, 2, figsize=(13, 5))

im = ax_corr.pcolormesh(lon2d, lat2d, corr.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax_corr, label="r")
ax_corr.set_title("Pixel-wise correlation (monthly, 2001–2021)")
ax_corr.set_xlabel("Longitude"); ax_corr.set_ylabel("Latitude")

lim = max(g_vals.max(), e_vals.max()) * 1.05
ax_scat.hexbin(g_vals, e_vals, gridsize=60, cmap="YlOrRd", mincnt=1, extent=[0, lim, 0, lim])
xfit = np.array([0, lim])
ax_scat.plot(xfit, xfit, "k--", lw=1, label="1:1")
ax_scat.plot(xfit, slope * xfit + intercept, "b-", lw=1.5,
             label=f"fit: y={slope:.2f}x+{intercept:.2f}")
ax_scat.set_xlim(0, lim); ax_scat.set_ylim(0, lim)
ax_scat.set_xlabel("GLEAM (mm day⁻¹)"); ax_scat.set_ylabel("FLUXCOM (mm day⁻¹)")
ax_scat.set_title("Daily spatial means (2001–2021)")
ax_scat.legend(fontsize=8)
ax_scat.text(0.05, 0.93, f"r={r:.3f}\nRMSE={rmse:.3f}\nBias={bias_scalar:+.3f}",
             transform=ax_scat.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

fig.suptitle("GLEAM vs FLUXCOM Evapotranspiration — Mediterranean (2001–2021)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_FIG}")
