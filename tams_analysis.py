#%%
import xarray as xr 
import tams
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
import os
from functions import dibujoMCS
from functions import dibujoCE
import seaborn as sns
from joblib import Parallel, delayed



import warnings
warnings.filterwarnings('ignore')
# %%
root = "./"
pr_pre = "precipitation_IMERG_2024_WA" # This can be changed to match the data and resolution.
data_path_pr = f"{root}TFM/{pr_pre}_0.25x0.25.nc"

temp_pre = "tb_merg_2024_WA" # This can be changed to match the data and resolution.
data_path_temp = f"{root}TFM/{temp_pre}_0.25x0.25.nc"

CHUNK_T = 96 

pr_data = (
    xr.open_dataset(data_path_pr,   chunks={"time": CHUNK_T, "lat": -1, "lon": -1})
      .transpose("time", "lat", "lon")
      .rename({"precipitation": "pr"})
)
temp_data = (
    xr.open_dataset(data_path_temp, chunks={"time": CHUNK_T, "lat": -1, "lon": -1})
      .rename({"Tb": "ctt"})
)
tbI = temp_data["ctt"]#.dropna(dim='time', how='all')
tp  = pr_data["pr"]


# %%
ds = xr.Dataset({'tb': tbI, 'tp': tp})
timeI = tbI.time.values.tolist()

cesI, _ = tams.identify(tbI.compute(), parallel=True);

# Añado variables de interés a los ces identificados:
def fun(ds,ce):
    if not ce.empty:
        ce = tams.data_in_contours(ds.tb, ce, agg=("mean", "min","max"), merge=True)
        ce = tams.data_in_contours(ds.tp, ce, agg=("mean","min", "max"), merge=True)
    return ce

cesI_all = Parallel(n_jobs=-2, verbose=10)(
    delayed(fun)(ds.isel(time=i).copy(deep=False),ce.copy())
    for i, ce in enumerate(cesI)
)



# %%
cesI_c = cesI_all.copy()
print(len(cesI_all), len(cesI_c))
print(True if cesI_all == cesI_c else False)


# %%
proj = -10
times = tbI.time.values.tolist()

ces_T = tams.track(cesI_c, times, u_projection=-10)


# %%
ce_clasI = tams.classify(ces_T)
# %%
dibujoMCS(ce_clasI)
# %%
dibujoCE(ce_clasI)
# %%
