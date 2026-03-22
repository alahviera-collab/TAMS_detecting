import pandas as pd
import geopandas as gpd
import os
import pickle as pkl
import logging

from .constants import _MCS_ORDER

def _ordered_cat(series : pd.Series):
    """Cast a string Series to the canonical ordered MCS category."""
    return pd.Categorical(series, categories=_MCS_ORDER, ordered=True)

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

def _load_classified(filename: str, root: str = "./TFM/pkl_sets") -> gpd.GeoDataFrame:
    """Load a classified GeoDataFrame from a pickle file."""
    filepath = os.path.join(root, f"{filename}.pkl")
    with open(filepath, "rb") as f:
        return pkl.load(f)

def _flat(df: pd.DataFrame):
    out = df.reset_index(level=["mcs_id", "itime"], drop=True).reset_index()
    out["str_class"] = _ordered_cat(out["str_class"].astype(str))
    return out