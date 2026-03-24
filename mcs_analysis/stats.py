import pandas as pd
import logging

# Relative imports
from .constants import _MCS_ORDER
from .utils import _add_spatial_temporal_cols, _ordered_cat

log = logging.getLogger(__name__)

def mcs_class_stats(cesC, has_tp: bool = True):
    """
    Calculates statistics for a classified GeoDataFrame from :func:`tams.classify`.

    Parameters
    ----------
    cesC : gpd.GeoDataFrame
        GeoDataFrame produced by ``tams.classify()``. Must contain columns:
        'mcs_class', 'area_km2', 'mean_tb', 'min_tb', 'max_tb', 'mcs_id',
        'itime', 'time', 'geometry'. 'mean_tp' is only required if has_tp=True.
    has_tp : bool
        Whether precipitation data is available. Default True.

    Returns
    -------
    dict with keys:
        - 'ce'        : full CE-level GeoDataFrame with derived columns added
        - 'ces_total' : Series — CE count per class
        - 'mcs_total' : Series — unique MCS count per class
        - 'mcs_area'  : DataFrame — total area per (mcs_id, itime, class)
        - 'mcs_tb'    : DataFrame — mean/min/max Tb per (mcs_id, itime, class)
        - 'mcs_tp'    : DataFrame or None — mean precip per (mcs_id, itime, class)
        - 'mcs_dur'   : DataFrame — duration [h] per (mcs_id, class)
    """

    # 0 - Define columns for later use
    ce  = _add_spatial_temporal_cols(cesC)
    ce["mcs_class"] = ce.mcs_class.cat.reorder_categories(_MCS_ORDER, ordered=True)
    ce["str_class"] = _ordered_cat(ce.mcs_class.astype(str))
    ce["tb_area"]   = ce["area_km2"] * ce["mean_tb"]
    if has_tp:
        ce["tp_area"] = ce["area_km2"] * ce["mean_tp"]

    # 1 - CE and MCS counts per class
    ces_total = ce.groupby(["mcs_class"], observed=True).mcs_id.count()
    mcs_total = ce.groupby(["mcs_class"], observed=True).mcs_id.nunique()

    logging.info(
        "CE percentages per type:\n%s",
        (ces_total / ces_total.sum() * 100).round(4)
    )
    logging.info(
        "MCS percentages per type:\n%s",
        (mcs_total / mcs_total.sum() * 100).round(4)
    )

    # 2 - Aggregation per (mcs_id, itime, class)
    grp = ce.groupby(["mcs_id", "itime", "str_class"])

    agg_dict = dict(
        area_km2=("area_km2", "sum"),
        tb_area =("tb_area",  "sum"),
        tb_min  =("min_tb",   "min"),
        tb_max  =("max_tb",   "max"),
    )
    if has_tp:
        agg_dict["tp_area"] = ("tp_area", "sum")

    agg = grp.agg(**agg_dict)

    # Weighted mean (handles MCS split across multiple CE fragments)
    agg["tb_mean"] = agg["tb_area"] / agg["area_km2"]
    if has_tp:
        agg["tp_mean"] = agg["tp_area"] / agg["area_km2"]

    # 3 - Duration per MCS
    dur = (
        ce.groupby(["mcs_id", "mcs_class"], observed=True)
        .time.agg(lambda s: (s.max() - s.min()).total_seconds() / 3600)
        .rename("duration_h")
        .dropna()
        .reset_index()
    )
    dur["str_class"] = _ordered_cat(dur["mcs_class"].astype(str))

    return {
        "ce"       : ce,
        "ces_total": ces_total,
        "mcs_total": mcs_total,
        "mcs_area" : agg[["area_km2"]].reset_index(),
        "mcs_tb"   : agg[["tb_mean", "tb_min", "tb_max"]].reset_index(),
        "mcs_tp"   : agg[["tp_mean"]].reset_index() if has_tp else None,
        "mcs_dur"  : dur,
    }