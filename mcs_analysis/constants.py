import geopandas as gpd

# Shared Constants 
_MCS_ORDER = ["DSL", "DLL", "CCC", "MCC"] # This follows Nunez Ocasio Classification
_EMPTY_GDF = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries([], dtype="geometry")
).set_crs(epsg=4326)
_KDE_KWARGS = dict(fill=False, alpha=0.6, common_norm=True)
_SAHEL_EXTENT = (lonmin, lonmax, latmin, latmax) = (-40,50,0,40) #(°W, °E, °S, °N)
