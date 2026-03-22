import pandas as pd
import logging

# Relative imports
from .constants import _MCS_ORDER
from .utils import _add_spatial_temporal_cols, _ordered_cat

log = logging.getLogger(__name__)

def mcs_class_stats(cesC):
    """
    Calculates statistics for a classified GeoDataFrame from :func:`tams.classify`.
 
    Parameters
    ----------
    cesC : gpd.GeoDataFrame
        GeoDataFrame produced by ``tams.classify()``. Must contain columns:
        'mcs_class', 'area_km2', 'mean_tb', 'min_tb', 'max_tb', 'mean_tp',
        'mcs_id', 'itime', 'time', 'geometry'.
 
    Returns
    -------
    dict with keys:
        - 'ce'        : full CE-level GeoDataFrame with derived columns added
        - 'ces_total' : Series — CE count per class
        - 'mcs_total' : Series — unique MCS count per class
        - 'mcs_area'  : DataFrame — total area per (mcs_id, itime, class)
        - 'mcs_tb'    : DataFrame — mean/min/max Tb per (mcs_id, itime, class)
        - 'mcs_tp'    : DataFrame — mean precip per (mcs_id, itime, class)
        - 'mcs_dur'   : DataFrame — duration [h] per (mcs_id, class)
    """

    # 0 - Define columns for later use
    ce  = _add_spatial_temporal_cols(cesC)
    ce["mcs_class"] = ce.mcs_class.cat.reorder_categories(_MCS_ORDER, ordered=True)
    ce["str_class"] = _ordered_cat(ce.mcs_class.astype(str))
    ce["tb_area"] = ce["area_km2"] * ce["mean_tb"]
    ce["tp_area"] = ce["area_km2"] * ce["mean_tp"]
 
    # 1 - order and get total number of ces_clases and mcs then calculate fraction of total for each class
    # Here, I count each cloud element that can or not be the same MCS.
    ces_total = ce.groupby(["mcs_class"], observed=True).mcs_id.count()
    # Here, I count each MCS.
    mcs_total = ce.groupby(["mcs_class"], observed=True).mcs_id.nunique()

    logging.info(
        "CE percentages per type:\n%s",
        (ces_total / ces_total.sum() * 100).round(4)
    )
    logging.info(
        "MCS percentages per type:\n%s",
        (mcs_total / mcs_total.sum() * 100).round(4)
    )

    # 2 - Group them and then get each stats for each class

    grp = ce.groupby(["mcs_id", "itime", "str_class"])
    agg = grp.agg(
        area_km2=("area_km2", "sum"),
        tb_area =("tb_area", "sum"),
        tb_min  =("min_tb", "min"),
        tb_max  =("max_tb", "max"),
        tp_area =("tp_area", "sum"),
    )

    # Do weighted mean for those cases where a single MCS is split across multiple CE fragements
    agg["tb_mean"] = agg["tb_area"]/agg["area_km2"]
    agg["tp_mean"] = agg["tp_area"]/agg["area_km2"]

    # 3 - estimate duration for each mcs
    dur = (
        ce.groupby(["mcs_id", "mcs_class"], observed=True)
        .time.agg(lambda s: (s.max() - s.min()).total_seconds() / 3600)
        .rename("duration_h")
        .dropna()
        .reset_index()
    )

    dur["str_class"] = _ordered_cat(dur["mcs_class"].astype(str))

    return {
        "ce" : ce,
        "ces_total" : ces_total,
        "mcs_total" : mcs_total,
        "mcs_area": agg[["area_km2"]].reset_index(),
        "mcs_tb": agg[["tb_mean", "tb_min", "tb_max"]].reset_index(),
        "mcs_tp": agg[["tp_mean"]].reset_index(),
        "mcs_dur": dur,
    }