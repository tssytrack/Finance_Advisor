#%% import packages
import numpy as np
import pandas as pd

#%% import data
map_directory = "/Users/dauku/Desktop/career/Cover Letter/Putnam/data/partnership_map.csv"
touchpoint_directory = "/Users/dauku/Desktop/career/Cover Letter/Putnam/data/touchpoints.csv"
trades_directory = "/Users/dauku/Desktop/career/Cover Letter/Putnam/data/trades.csv"

map = pd.read_csv(map_directory)
touchpoint = pd.read_csv(touchpoint_directory)
trades = pd.read_csv(trades_directory)

#%% data quality check
# Missing values
print(map.isna().sum())
print(touchpoint.isna().sum())
print(trades.isna().sum()) # only trades contains missing data

# data type
print(map.dtypes)
print(touchpoint.dtypes)
print(trades.dtypes)

#%% impute missing values in trades table

# ELITE_FIRM
print(trades.ELITE_FIRM.unique()) # this shows missing means no
trades["ELITE_FIRM"].fillna("NO", inplace = True)

