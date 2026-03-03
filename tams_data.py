import xarray as xr 
import tams
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

root = "./"
pr_pre = "precipitation_IMERG_1999_WA" # This can be changed to match the data and resolution.
data_path_pr = f"{root}TFM/{pr_pre}_0.25x0.25.nc"

temp_pre = "tb_merg_1999_WA" # This can be changed to match the data and resolution.
data_path_temp = f"{root}TFM/{temp_pre}_0.1x0.1.nc"

pr_data = xr.open_dataset(data_path_pr)
pr_data = pr_data.transpose("time", "lat", "lon") # In order to put dims correctly time,lat,lon.
temp_data = xr.open_dataset(data_path_temp)

ds_full = xr.merge([temp_data, pr_data])
ds_full = ds_full.rename({"Tb":"ctt","precipitation":"pr"})