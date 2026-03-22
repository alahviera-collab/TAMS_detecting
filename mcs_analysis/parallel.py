import xarray as xr 
import geopandas as gpd
import pandas as pd
import os
import tams
import logging
from joblib import Parallel, delayed
from pathlib import Path

# Relative imports
from .constants import _EMPTY_GDF
from .utils import _make_empty_like

log = logging.getLogger(__name__)

def open_pr_tp(root_data: str, pr_file: str, tb_file: str, CHUNK_T: float = 96, isnan: bool = True) -> tuple:
    """
    Find precipitation and brightness temperature. Using *chunks* with dask array, returns both variables as *tbI* and *tp*.
    precipitation and temperature are renamed into pr and ctt respectively for tams usage.

    Parameters
    ----------------
    root_data:
        Path as *str* where data is stored.
    pr_file:
        Name of precipitation file in nc format. -> Gets transpose as lat and lon are swaped in dimention.
    tb_path:
        Name of brightness temperature file in nc format.
    CHUNK_T:
        96 by default, is used to make dask arrays.
    isnan:
        Default True, this is used for those cases where (normally tb) has nan values so we drop them

    Returns
    ---------------
    tuple
    """

    pr_path = Path(root_data) / pr_file
    tb_path = Path(root_data) / tb_file

    pr_data = (
        xr.open_dataset(pr_path, chunks={"time": CHUNK_T, "lat": -1, "lon": -1})
            .transpose("time", "lat", "lon")
            .rename({"precipitation": "pr"})
    )

    tb_data = (
        xr.open_dataset(tb_path, chunks={"time": CHUNK_T, "lat": -1, "lon": -1})
            .rename({"Tb": "ctt"})
    )

    if isnan:
        # Drop based on tb only — tb is the critical variable and the most
        # likely source of NaN frames (missing satellite passes).
        tbI = tb_data["ctt"].dropna(dim='time', how='all')

        # Reindex tp to tb's valid time axis so both share the same timestamps.
        # tp gaps are filled with 0 (no precip), not dropped independently.
        tp = pr_data["pr"].sel(time=tbI.time).fillna(0)
    else:
        tbI = tb_data["ctt"]
        tp  = pr_data["pr"]

    return tp, tbI

def enrich_ce_slice_nan(ds_slice: xr.Dataset, ce: gpd.GeoDataFrame, has_tp: bool = True):
    """
    Enrich a CE GeoDataFrame with tb/tp statistics for datasets with Valid tb frame, but no MCS contours detected by
    timestamps. Empty CEs and ValueError are silently skipped by returning
    an empty GeoDataFrame. Then runs data_in_contours from tams into ce
    ce are elements inside of cesI (result of tams.identify)
    In order to drop non MCS timestamps, we use an EMPTY gpd.GeoDataFrame
    """ 

    if ce.empty:
        return _EMPTY_GDF
    try:
        tb_slice = ds_slice.tb.compute()
        ce = tams.data_in_contours(tb_slice, ce, agg=("mean", "min","max"), merge=True)

        if has_tp:
            tp_slice = ds_slice.tp.compute().fillna(0)
            ce = tams.data_in_contours(tp_slice, ce, agg=("mean", "min","max"), merge=True)

    except ValueError:
        log.warning("data_in_contours failed for time slice; returning empty name.")
        return _EMPTY_GDF # Return empty frame.
    
    return ce

def enrich_ce_slice(
    ds_slice: xr.Dataset,
    ce: gpd.GeoDataFrame,
    has_tp: bool = True,
) -> gpd.GeoDataFrame:
    """
    Enrich a CE GeoDataFrame with tb/tp statistics for datasets without
    NaN timestamps. If an empty CE is found, raises a ValueError suggesting
    to use enrich_ce_slice_nan() instead.
    """
    if ce.empty:
        return ce
    
    tb = ds_slice.tb.compute()
    ce = tams.data_in_contours(tb, ce, agg=("mean", "min", "max"), merge=True)

    if has_tp:
        tp = ds_slice.tp.compute()
        ce = tams.data_in_contours(tp, ce, agg=("mean", "min", "max"), merge=True)

    return ce

def Parallel_nan_ds(cesI, ds_pre, has_tp:bool = True):
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
        delayed(enrich_ce_slice_nan)(ds_pre.isel(time=i), ce.copy(), has_tp)
        for i, ce in enumerate(cesI)
    )

def Parallel_ds(
    cesI: list[gpd.GeoDataFrame],
    ds_pre: xr.Dataset,
    has_tp:bool = True,
) -> list[gpd.GeoDataFrame]:
    """Parallel enrichment for datasets without NaN timestamps."""
    return Parallel(n_jobs=-2, verbose=10)(
        delayed(enrich_ce_slice)(ds_pre.isel(time=i), ce.copy(), has_tp)
        for i, ce in enumerate(cesI)
    )

def save_parquet(cesI, filename: str, root: str = "./TFM/parquet_sets"):

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
        does not exist. Defaults to ``'./TFM/parquet_sets'``.
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