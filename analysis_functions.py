import xarray as xr 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
import os
import seaborn as sns
import numpy as np
import tams
import logging

from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
from typing import Optional


log = logging.getLogger(__name__)

# Shared Constants 
_MCS_ORDER = ["DSL", "DLL", "CCC", "MCC"] # This follows Nunez Ocasio Classification
_EMPTY_GDF = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries([], dtype="geometry")
).set_crs(epsg=4326)
_KDE_KWARGS = dict(fill=False, alpha=0.6, common_norm=True)
_SAHEL_EXTENT = (lonmin, lonmax, latmin, latmax) = (-40,50,0,40) #(°W, °E, °S, °N)

# Shared functions
def _ordered_cat(series : pd.Series):
    """Cast a string Series to the canonical ordered MCS category."""
    return pd.Categorical(series, categories=_MCS_ORDER, ordered=True)

def _boxplot_by_class(df: pd.DataFrame, col:str, ylabel:str, ax):
    """
    Generic  helper function: plot a boxplot of *col* grouped by 'str_class' on *ax*.
    Expects df to have an ordered-categorical 'str_class' column.
    """

    df.boxplot(column=col, by='str_class', ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

def _make_empty_like(reference: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame with the same columns as *reference*."""
    cols = [c for c in reference.columns if c != "time"]
    return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")

def _add_spatial_temporal_cols(ce: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Derive cent_lat, cent_lon, and day_yr from geometry and time.
    Returns a copy with the new columns added.
    """
    ce = ce.copy()
    centroids = ce.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    ce["cent_lat"] = centroids.y
    ce["cent_lon"] = centroids.x
    ce["day_yr"]   = ce["time"].dt.dayofyear
    return ce

def _kde_scatter(ax, data:pd.DataFrame, x: str, y:str, clip=None):
    """Auxiliar funciton - Scatter + KDE overlay on a given axis."""

    ax.plot(data[x], data[y], ".", ms=1, mec="none", mfc="0.35", alpha=0.9)
    sns.kdeplot(
        x=x, y=y,
        color= sns.color_palette()[0],
        clip=clip,
        ax=ax,
        data=data,
        **_KDE_KWARGS,
    )


#-----------------------------
# Axuliar functions for reading data

def enrich_ce_slice_nan(ds_slice: xr.Dataset, ce: gpd.GeoDataFrame):
    """
    Enrich a CE GeoDataFrame with tb/tp statistics for datasets with NaN
    timestamps. Empty CEs and ValueError are silently skipped by returning
    an empty GeoDataFrame. Then runs data_in_contours from tams into ce
    ce are elements inside of cesI (result of tams.identify)
    In order to drop nan timestamps, we use an EMPTY gpd.GeoDataFrame
    """ 

    if ce.empty:
        return _EMPTY_GDF
    try:
        tb_slice = ds_slice.tb.compute()
        tp_slice = ds_slice.tp.compute().fillna(0)
        ce = tams.data_in_contours(tb_slice, ce, agg=("mean", "min","max"), merge=True)
        ce = tams.data_in_contours(tp_slice, ce, agg=("mean", "min","max"), merge=True)
    except ValueError:
        log.warning("data_in_contours failed for time slice; returning empty name.")
        return _EMPTY_GDF # Return empty frame.
    
    return ce

def enrich_ce_slice(
    ds_slice: xr.Dataset,
    ce: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Enrich a CE GeoDataFrame with tb/tp statistics for datasets without
    NaN timestamps. If an empty CE is found, raises a ValueError suggesting
    to use enrich_ce_slice_nan() instead.
    """
    if ce.empty:
        return ce
    tb = ds_slice.tb.compute()
    tp = ds_slice.tp.compute()
    ce = tams.data_in_contours(tb, ce, agg=("mean", "min", "max"), merge=True)
    ce = tams.data_in_contours(tp, ce, agg=("mean", "min", "max"), merge=True)
    return ce

def Parallel_nan_ds(cesI, ds_pre):
    """
    This is main excution to drop nan timestamps using :func:`enrich_ce_slice`.
    Please notice that cesI and dr_pre are made by tams.identify() and xr.Dataset type that contains tb and tp, respectively.

    Parameters
    ----------
    cesI:
        List of CE GeoDataFrame (from :func:`tams.identify`)
    ds_pre:
        DataSet that contains brightness temperature and total precipitation.
    """
    return Parallel(n_jobs=-2, verbose=10)(
        delayed(enrich_ce_slice_nan)(ds_pre.isel(time=i), ce.copy())
        for i, ce in enumerate(cesI)
    )

def Parallel_ds(
    cesI: list[gpd.GeoDataFrame],
    ds_pre: xr.Dataset,

) -> list[gpd.GeoDataFrame]:
    """Parallel enrichment for datasets without NaN timestamps."""
    return Parallel(n_jobs=-2, verbose=10)(
        delayed(enrich_ce_slice)(ds_pre.isel(time=i), ce.copy())
        for i, ce in enumerate(cesI)
    )

def save_parquet(cesI, filename: str, root: str = "./parquet_sets"):

    """
    Concatenate *cesI* and persist it as a single Parquet file.
 
    Parameters
    ----------
    cesI:
        List of CE GeoDataFrames (e.g. the result of :func:`Parallel_nan_ds`).
    filename:
        Output file name, without extension.
    root:
        Directory where the file will be written. Created automatically if it
        does not exist. Defaults to ``'./parquet_sets'``.
    """
 
    os.makedirs(root, exist_ok=True)
    filepath = os.path.join(root, f"{filename}.parquet")

    frames = []
    for i, r in enumerate(cesI):
        if not r.empty:
            r = r.copy()
            r["itime"] = i        # ← record time step index before concatenating
            frames.append(r)

    ces_master = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry"
    ).set_crs(epsg=4326)

    ces_master.to_parquet(filepath)

    log.info("Saved %d rows to %s", len(ces_master), filepath)

def list_from_parquet(
    par_file: str,
    n_times: int,
) -> list[gpd.GeoDataFrame]:
    """
    Reconstruct a per-timestep list from a Parquet file that was saved with
    NaN timestamps skipped (i.e. some itime indices are absent).

    Missing indices are filled with empty GeoDataFrames so the returned list
    aligns with the original time axis and can be passed to tams.track().

    Parameters
    ----------
    par_file : str
        Path to the .parquet file produced by save_parquet().
    n_times : int
        Total number of time steps in the original time axis
        (e.g. len(tbI.time)).
    """
    if not isinstance(par_file, str):
        raise TypeError(
            f"par_file must be a file path string, got {type(par_file).__name__!r}. "
            "Did you pass the GeoDataFrame instead of the path?"
        )
    
    ces_par = gpd.read_parquet(par_file)
    present = set(ces_par["itime"].unique())

    return [
    gpd.GeoDataFrame(
        ces_par[ces_par["itime"] == i].drop(columns="itime").reset_index(drop=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    if i in present
    else _make_empty_like(ces_par)
    for i in range(n_times)
    ]


#------------------------------
def dibujoMCS(ce_clasI):
    """
    Distribution plots for CEs and MCSs broken down by class.
    """
    # ¿Cómo se distribuyen los CE y los MCS en las distintas clases?
    fig = plt.figure(figsize = (10,20))
    ncol,nfil = 6, 2
    igraf=0

    
    ce = _add_spatial_temporal_cols(ce_clasI)
    ce["mcs_class"]  = ce.mcs_class.cat.reorder_categories(_MCS_ORDER, ordered=True)
    ce["str_class"]  = _ordered_cat(ce.mcs_class.astype(str))

    
    # igraf = igraf + 1
    # PANEL 1: Distribución de CEs (para cada tiempo, los elementos son disjuntos y solo los clasifico y cuento)
    ces_clases = ce.groupby(["mcs_class"], observed=True).mcs_id.count()
    ax = fig.add_subplot(ncol,nfil, 1)
    ax.bar(ces_clases.index, ces_clases/ces_clases.sum()*100)
    ax.grid()
    ax.set_title('Count of CEs per type. Total='+str(ces_clases.sum()))
    ax.set_ylabel('[%]')

    #igraf = igraf + 1
    # PANEL 2: Distribución de MCSs (cuento cada identificador mcs_id una sola vez)
    mcs_clases=ce.groupby(["mcs_class"], observed=True).mcs_id.nunique()
    ax = fig.add_subplot(ncol,nfil,2)
    ax.bar(mcs_clases.index, mcs_clases/mcs_clases.sum()*100)
    ax.grid()
    ax.set_title('Count of MCSs per type. Total='+str(mcs_clases.sum()))
    ax.set_ylabel('[%]')
    
    #igraf = igraf + 1
    #PANEL 3 AL 5: Localización en latitud, longitud y día del año típica por CE (por MCS sería muy complicado hacerlo bien)
    for igraf, (col, ylabel) in enumerate(
        [("cent_lat", "Mean latitude"), ("cent_lon", "Mean longitude"), ("day_yr", "Timing in the year")],
        start=3,
    ):
        ax = fig.add_subplot(ncol, nfil, igraf)
        ce.boxplot(column=col, by="mcs_class", ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")

    #igraf = igraf + 1
    # PANEL 6 AL 10: Boxplot del área de cada MCS clasificada por tipo 
    # (para cada tiempo, sumo el área del mismo MCS)
    ce["tb_area"]= ce["area_km2"] * ce["mean_tb"]
    ce["tp_area"] = ce["area_km2"] * ce["mean_tp"]

    grp = ce.groupby(["mcs_id", "itime", "str_class"])

    agg = grp.agg(
        area_km2=("area_km2","sum"),
        tb_area=("tb_area", "sum"),
        tp_area=("tp_area", "sum"),
        min_tb=("min_tb", "sum"),
    )
    agg["mean_tb"]= agg["tb_area"]/agg["area_km2"]
    agg["mean_tp"]= agg["tp_area"]/agg["area_km2"]

    dur = (
        ce.groupby(["mcs_id", "mcs_class"], observed=True)
        .time.agg(lambda s: (s.max() - s.min()).total_seconds() / 3600)
        .rename("duration_h")
        .dropna()
        .reset_index()
    )

    dur["mcs_class"] = _ordered_cat(dur["mcs_class"].astype(str))

    def _flat(df: pd.DataFrame):
        out = df.reset_index(level=["mcs_id", "itime"], drop=True).reset_index()
        out["str_class"]  = _ordered_cat(out["str_class"].astype(str))
        return out
    
    flat = _flat(agg)

    # PANEL 6: total area
    ax = fig.add_subplot(ncol, nfil, 6)
    _boxplot_by_class(flat, "area_km2", "Total area [km$^2]", ax)
    ax.set_yscale("log")


    #igraf = igraf + 1
    # PANEL 7: Duración total MCS
    ax = fig.add_subplot(ncol,nfil,7)
    dur.boxplot(column="duration_h", by="mcs_class", ax=ax)
    ax.set_ylabel("Duration [h]")
    ax.set_xlabel("")

    
    #igraf = igraf + 1
    # PANEL 8: temperatura brillo promedio en cada MCSs clasificado por tipo
    # (para cada tiempo, sumo la precipitación media pesada por el área abarcada)
    ax = fig.add_subplot(ncol, nfil, 8)
    _boxplot_by_class(flat, "mean_tb", "Mean Tb [K]", ax)
    
    
    #igraf = igraf + 1
    # PANEL 9: Temperatura de brillo mínima por MCS:
    ax = fig.add_subplot(ncol, nfil, 9)
    _boxplot_by_class(flat, "min_tb", "Minum Tb [K]", ax)

    #igraf = igraf + 1
    # PANEL 10: Precipitación promedio en cada MCSs clasificado por tipo
    # (para cada tiempo, sumo la precipitación media pesada por el área abarcada)
    ax = fig.add_subplot(ncol, nfil, 10)
    _boxplot_by_class(flat, "mean_tp", "Mean precip [mm h${-1}$]",ax)
    
    
def dibujoCE(ce_trackI):
    """
    Kernel density estimation (KDE) for mcs and various plots.
    """

    fig = plt.figure(figsize = (10,8))
    
    n = ce_trackI.mcs_id.nunique()
    gb = ce_trackI.groupby("mcs_id")
    
    mcs_stats = gb.agg(
        arrea_max=("area_km2", "max"),
        min_tb=("min_tb", "min"),
        mean_tp=("mean_tp", "mean"),
        max_tp=("max_tp", "max"),
        time_min=("time", "min"),
        time_max=("time", "max")
    )

    mcs_stats["duration_h"] = (
        (mcs_stats["time_max"] - mcs_stats["time_min"]).dt.total_seconds() / 3600
    )

    mcs_stats = mcs_stats.astype({
        "area_max": float,
        "min_tb": float,
        "mean_tp": float,
        "min_tp": float,
    })

    mcs_area_max = ce_trackI.groupby(["mcs_id", "itime"]).area_km2.sum().groupby("mcs_id").max()

    # PANEL 1: KDE of max area
    ax = fig.add_subplot(2, 2, 1)
    mcs_stats["area_max"].plot.kde(ax=ax, label="CE")
    mcs_area_max.plot.kde(ax=ax, label="MCS")
    ax.text(0.99, 0.03, f"$N={n}$", size=9, ha="right", va="bottom", transform=ax.transAxes)
    ax.set_xlim(xmin=1000)
    ax.set_xscale("log")
    ax.set_ylim(ymin=0)
    ax.legend(loc="center right")
    ax.set_xlabel("Max area [km$^2$]")
    ax.grid()

    #ax.set_title('Área máxima cubierta por MCS')
    #print('Hay que mirar cómo es en observaciones y cómo varía con distintos umbrales para T')
    
    
    ###################################################################
    # PANEL 2: Distribución de duración del MCS frente a área máxima cubierta en un tiempo dado
    
    ax = fig.add_subplot(2,2,2)
    data_ad = mcs_stats[["area_max", "duration_h"]].dropna()
    kde = gaussian_kde((data_ad.values.T))
    X, Y = np.mgrid[0:5e6:100j, 0:30:100j]
    pos = np.column_stack((X.ravel(), Y.ravel())).T
    Z = np.reshape(kde(pos).T, X.shape)
    im = ax.contourf(
        X, Y, 
        np.log10(np.where(Z > 0, Z, np.nan)),
        levels=np.linspace(np.log10(5e-11), np.log10(5e-7), 19),
        extend="max",
        cmap=sns.color_palette("Blues", as_cmap=True),
    )
    
    ax.text(0.99, 0.03, f"$N={n}$", size=9, ha="right", va="bottom", transform=ax.transAxes)
    ax.set_ylabel('Duration [h]')
    ax.set_xlabel('Máx area [km$^2$]')
    
    ###################################################################
    # PANEL 3: Comparación de min Tb vs media de precipitación.

    ax = fig.add_subplot(2,2,3)
    data1 = mcs_stats[["min_tb", "mean_tp"]].dropna()  
    ax.plot(data1["min_tb"], data1["mean_tp"], ".", mec="none", mfc="0.35", alpha=0.9)
    sns.kdeplot(
        x="min_tb", y="mean_tp",
        fill=False, color=sns.color_palette()[0], alpha=0.6,
        common_norm=True, clip=(0, None),
        ax=ax, data=data1,
    )

    
    ###################################################################
    # PANEL 4: Comparación de min Tb vs max de precipitación.

    ax = fig.add_subplot(2,2,4)
    data2 = mcs_stats[["min_tb", "max_tp"]].dropna()  
    ax.plot(data2["min_tb"], data1["max_tp"], ".", mec="none", mfc="0.35", alpha=0.9)
    sns.kdeplot(
        x="min_tb", y="max_tp",
        fill=False, color=sns.color_palette()[0], alpha=0.6,
        common_norm=True, clip=(0, None),
        ax=ax, data=data2,
    )


def seasonal_latlon(
        tbI: xr.DataArray,
        ce_clasI: gpd.GeoDataFrame,
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
    ce_clasI : gpd.GeoDataFrame
        GeoDataFrame produced by ``tams.classify()``. Must contain columns:
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

    ce = _add_spatial_temporal_cols(ce_clasI[ce_clasI["mcs_class"] == mcs_class])

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

