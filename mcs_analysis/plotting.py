import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy.stats import gaussian_kde

# Relative imports
from .constants import _MCS_ORDER, _SAHEL_EXTENT, _KDE_KWARGS
from .stats import mcs_class_stats

def _boxplot_by_class(df: pd.DataFrame, col: str, ylabel: str, ax):
    """
    Generic helper function: plot a boxplot of *col* grouped by 'str_class' on *ax*.
    Expects df to have an ordered-categorical 'str_class' column.
    """
    df.boxplot(column=col, by='str_class', ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

def _kde_scatter(ax, data: pd.DataFrame, x: str, y: str, clip=None):
    """Auxiliar funciton - Scatter + KDE overlay on a given axis."""
    ax.plot(data[x], data[y], ".", ms=1, mec="none", mfc="0.35", alpha=0.9)
    sns.kdeplot(
        x=x, y=y,
        color=sns.color_palette()[0],
        clip=clip,
        ax=ax,
        data=data,
        **_KDE_KWARGS,
    )

def dibujoMCS(cesC: gpd.GeoDataFrame, has_tp: bool = True):
    """
    Distribution plots for CEs and MCSs broken down by class.
    """
    stats = mcs_class_stats(cesC, has_tp=has_tp)

    ce         = stats["ce"]
    ces_clases = stats["ces_total"]
    mcs_clases = stats["mcs_total"]

    # Adjust grid: 5 rows if no tp (drop panel 10), 6 rows if tp present
    nrow = 6 if has_tp else 5
    ncol = 2
    fig  = plt.figure(figsize=(10, nrow * 10 / 3))

    # PANEL 1: CE count per class
    ax = fig.add_subplot(nrow, ncol, 1)
    ax.bar(ces_clases.index, ces_clases / ces_clases.sum() * 100)
    ax.grid()
    ax.set_title("Count of CEs per type. Total=" + str(ces_clases.sum()))
    ax.set_ylabel("[%]")

    # PANEL 2: MCS count per class
    ax = fig.add_subplot(nrow, ncol, 2)
    ax.bar(mcs_clases.index, mcs_clases / mcs_clases.sum() * 100)
    ax.grid()
    ax.set_title("Count of MCSs per type. Total=" + str(mcs_clases.sum()))
    ax.set_ylabel("[%]")

    # PANELS 3-5: Spatial/temporal location by CE
    for igraf, (col, ylabel) in enumerate(
        [("cent_lat", "Mean latitude"), ("cent_lon", "Mean longitude"), ("day_yr", "Timing in the year")],
        start=3,
    ):
        ax = fig.add_subplot(nrow, ncol, igraf)
        ce.boxplot(column=col, by="mcs_class", ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")

    # PANEL 6: Total area per MCS
    ax = fig.add_subplot(nrow, ncol, 6)
    _boxplot_by_class(stats["mcs_area"], "area_km2", "Total area [km$^2$]", ax)
    ax.set_yscale("log")

    # PANEL 7: Duration per MCS
    ax = fig.add_subplot(nrow, ncol, 7)
    _boxplot_by_class(stats["mcs_dur"], "duration_h", "Duration [h]", ax)

    # PANEL 8: Mean Tb per MCS
    ax = fig.add_subplot(nrow, ncol, 8)
    _boxplot_by_class(stats["mcs_tb"], "tb_mean", "Mean Tb [K]", ax)

    # PANEL 9: Min Tb per MCS
    ax = fig.add_subplot(nrow, ncol, 9)
    _boxplot_by_class(stats["mcs_tb"], "tb_min", "Minimum Tb [K]", ax)

    # PANEL 10: Mean precipitation — only when tp is available
    if has_tp:
        ax = fig.add_subplot(nrow, ncol, 10)
        _boxplot_by_class(stats["mcs_tp"], "tp_mean", "Mean precip [mm h$^{-1}$]", ax)

    fig.tight_layout()
    return fig

def dibujoCE(ce_trackI: gpd.GeoDataFrame, has_tp: bool = True):
    """
    Kernel density estimation (KDE) for MCSs and various plots.
    """
    # Adjust grid: 2 rows if no tp (drop panels 3 & 4), 2x2 if tp present
    nrow = 2 if has_tp else 1
    fig  = plt.figure(figsize=(10, 4 * nrow))

    n  = ce_trackI.mcs_id.nunique()
    gb = ce_trackI.groupby("mcs_id")

    # Build aggregation dict — tp columns only when available
    agg_dict = dict(
        area_max =("area_km2", "max"),
        min_tb   =("min_tb",   "min"),
        mean_tb  =("mean_tb",  "mean"),
        time_min =("time",     "min"),
        time_max =("time",     "max"),
    )
    if has_tp:
        agg_dict.update(
            mean_tp=("mean_tp", "mean"),
            max_tp =("max_tp",  "max"),
        )

    mcs_stats = gb.agg(**agg_dict)
    mcs_stats["duration_h"] = (
        (mcs_stats["time_max"] - mcs_stats["time_min"]).dt.total_seconds() / 3600
    )

    cast_dict = {"area_max": float, "min_tb": float}
    if has_tp:
        cast_dict.update({"mean_tp": float, "max_tp": float})
    mcs_stats = mcs_stats.astype(cast_dict)

    mcs_area_max = ce_trackI.groupby(["mcs_id", "itime"]).area_km2.sum().groupby("mcs_id").max()

    # PANEL 1: KDE of max area
    ax = fig.add_subplot(nrow, 2, 1)
    mcs_stats["area_max"].plot.kde(ax=ax, label="CE")
    mcs_area_max.plot.kde(ax=ax, label="MCS")
    ax.text(0.99, 0.03, f"$N={n}$", size=9, ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xlim(xmin=1000)
    ax.set_xscale("log")
    ax.set_ylim(ymin=0)
    ax.legend(loc="center right")
    ax.set_xlabel("Max area [km$^2$]")
    ax.grid()

    # PANEL 2: KDE of duration vs max area
    ax = fig.add_subplot(nrow, 2, 2)
    data_ad = mcs_stats[["area_max", "duration_h"]].dropna()
    kde = gaussian_kde(data_ad.values.T)
    X, Y = np.mgrid[0:5e6:100j, 0:30:100j]
    pos  = np.column_stack((X.ravel(), Y.ravel())).T
    Z    = np.reshape(kde(pos).T, X.shape)
    ax.contourf(
        X, Y,
        np.log10(np.where(Z > 0, Z, np.nan)),
        levels=np.linspace(np.log10(5e-11), np.log10(5e-7), 19),
        extend="max",
        cmap=sns.color_palette("Blues", as_cmap=True),
    )
    ax.text(0.99, 0.03, f"$N={n}$", size=9, ha="right", va="bottom", transform=ax.transAxes)
    ax.set_ylabel("Duration [h]")
    ax.set_xlabel("Max area [km$^2$]")

    # PANELS 3 & 4: min Tb vs precipitation — only when tp is available
    if has_tp:
        # PANEL 3: min Tb vs mean precipitation
        ax = fig.add_subplot(nrow, 2, 3)
        data1 = mcs_stats[["min_tb", "mean_tp"]].dropna()
        ax.plot(data1["min_tb"], data1["mean_tp"], ".", mec="none", mfc="0.35", alpha=0.9)
        sns.kdeplot(
            x="min_tb", y="mean_tp",
            fill=False, color=sns.color_palette()[0], alpha=0.6,
            common_norm=True, clip=(0, None),
            ax=ax, data=data1,
        )

        # PANEL 4: min Tb vs max precipitation
        ax = fig.add_subplot(nrow, 2, 4)
        data2 = mcs_stats[["min_tb", "max_tp"]].dropna()
        ax.plot(data2["min_tb"], data2["max_tp"], ".", mec="none", mfc="0.35", alpha=0.9)
        sns.kdeplot(
            x="min_tb", y="max_tp",
            fill=False, color=sns.color_palette()[0], alpha=0.6,
            common_norm=True, clip=(0, None),
            ax=ax, data=data2,
        )

    fig.tight_layout()
    return fig

def seasonal_latlon(
        tbI: xr.DataArray,
        stats: dict,
        mcs_class: str,
) -> plt.Figure:
    """
    Plot the seasonal (day-of-year) relationship between MCS location for a
    given MCS class, across three panels:
 
      1. Day-of-year vs latitude
      2. Longitude vs day-of-year
      3. Map: longitude vs latitude with coastlines
 
    The map extent is inferred automatically from the data, with an optional
    padding applied on each side.
 
    Parameters
    ----------
    tbI : xr.DataArray
        Brightness temperature array with a 'time' coordinate in datetime
        format. Used only to derive the season boundaries (first/last day
        of year).
    ce_clasI : dict
        dict produced by ``mcs_class_stats``. Must contain columns:
        'mcs_class', 'cent_lat', 'cent_lon', 'day_yr', 'mcs_id', 'itime'.
    mcs_class : str
        MCS class to plot. Must be one of 'DSL', 'DLL', 'CCC', 'MCC'.
 
    Returns
    -------
    matplotlib.figure.Figure
    """

    if mcs_class not in _MCS_ORDER:
        raise ValueError(f"mcs_class must be one of {_MCS_ORDER}, got {mcs_class!r}")
    
    start = int(tbI.time.min().dt.dayofyear)
    end = int(tbI.time.max().dt.dayofyear)

    ce = stats["ce"]
    ce = ce[ce["str_class"] == mcs_class]

    if ce.empty:
        raise ValueError(f"No data found for mcs_class={mcs_class!r}.")
    
    gb = ce.groupby(['mcs_id', 'itime'])
    stats = pd.DataFrame({
        "day_yr": gb.day_yr.mean().astype(float),
        "cent_lat": gb.cent_lat.mean().astype(float),
        "cent_lon": gb.cent_lon.mean().astype(float) + 360,
    })

    lonmin, lonmax, latmin, latmax = _SAHEL_EXTENT
 
    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"Seasonal lat/lon distribution — {mcs_class}", y=1.01)
 
    # Panel 1: day-of-year vs latitude
    ax1 = fig.add_subplot(2, 2, 1)
    _kde_scatter(ax1, stats, x="day_yr", y="cent_lat", clip=(0, None))
    ylim = ax1.get_ylim()
    ax1.vlines([start, end], *ylim, colors="orange", label="Season bounds")
    ax1.set_ylim(ylim)
    ax1.set_xlabel("Day of year")
    ax1.set_ylabel("Centroids with average latitudes")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
 
    # Panel 2: longitude vs day-of-year
    ax2 = fig.add_subplot(2, 2, 2)
    _kde_scatter(ax2, stats, x="cent_lon", y="day_yr", clip=(0, None))
    xlim = ax2.get_xlim()
    ax2.hlines([start, end], *xlim, colors="orange", label="Season bounds")
    ax2.set_xlim(xlim)
    ax2.set_xlabel("Centroids with average longitudes")
    ax2.set_ylabel("Day of year")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
 
    # Panel 3: map
    ax3 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    stats_map = stats.copy()
    stats_map["cent_lon"] -= 360  # restore -180/180 for PlateCarree
 
    _kde_scatter(ax3, stats_map, x="cent_lon", y="cent_lat")
    ax3.coastlines(linewidth=2)
    ax3.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
    ax3.set_xlabel("Longitude [°]")
    ax3.set_ylabel("Latitude [°]")
 
    fig.tight_layout()