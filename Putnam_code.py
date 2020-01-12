#%% import packages
import numpy as np
import pandas as pd
import re

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

# Imputing trades table

# ELITE_FIRM
print(trades.ELITE_FIRM.unique()) # this shows missing means no
trades["ELITE_FIRM"].fillna("NO", inplace = True)

# data type
print(map.dtypes)
print(touchpoint.dtypes) # DATE should be in the type of date
print(trades.dtypes) # TRAN_ID and PROD_ID should be nominal and TRAN_DATE should be date

touchpoint["DATE"] = pd.to_datetime(touchpoint.DATE)
trades["TRAN_DATE"] = pd.to_datetime(trades.TRAN_DATE)
trades["TRAN_ID"] = trades.TRAN_ID.astype("object")
trades["PROD_ID"] = trades.PROD_ID.astype("object")

#%% EDA
# Explore partners and individual adviors
map["individual"] = map.INDIVIDUAL_IDS.str.replace(r"[\]\[\'\s]+", "")  # using regular expression to get rid of square brackets and qutations
map["individual1"] = map.apply(lambda x: x[2].split(","), axis = 1)  # split the string based on comma and store it as a list
map["length"] = map.apply(lambda x: len(x[3]), axis = 1)
partner_individual2 = map.explode("individual1")
partner_individual2.drop(["individual", "INDIVIDUAL_IDS"], axis = 1, inplace = True)
partner_individual2.columns = ["partnership", "individuals", "length"]
partner_individual = map["individual1"].apply(pd.Series)  # span all the list into columns in the dataframe
partner_individual["Partner"] = map["PARTNERSHIP_ID"]
col_names = ["member1", "member2", "member3", "member4", "member5", "member6", "member7", "member8",
             "member9", "member10", "member11", "member12", "member13", "member14", "member15", "member16",
             "member17", "member18", "member19", "member20", "Partner"]
partner_individual.columns = col_names  # rename the columns
col_name_order = [col_names[-1]] + list(col_names[:-1])  # reorder the columns
partner_individual = partner_individual[col_name_order]  # reorder the columns

partner_list = partner_individual["Partner"].values  # get a list of partners
individual_list = pd.unique(partner_individual.iloc[:, 1:].values.ravel("K")) # get a list of individual advisors

# checking whether the ids in touchpoint table are from partner or individuals
id_touchpoint = touchpoint.ID.values
touchpoint_inter = np.intersect1d(id_touchpoint, partner_list) # empty collection implies that all the contacts made in touchpoint table are made by indifidual advisors

# Checking whether advisors can be in multiple partnerships
print(partner_individual[partner_individual.values == individual_list[0]].shape)  # an example shows that advisors can be in multiple partnership

# Divide trade into the trade made by individuals or partners
individuals = trades[trades["TRADE_TYPE"] == "I"]
partners = trades[trades["TRADE_TYPE"] == "P"]

# Aggregate trade amount to each partner or each individual
# ind_agg = individuals.groupby(["ID"])["TRAN_AMT"].sum()
# ind_agg.sort_values(ascending = False, inplace = True)
# part_agg = partners.groupby(["ID"])["TRAN_AMT"].sum()
# part_agg.sort_values(ascending = False, inplace = True)

# distribute each partnership's trade into each individual advisors
merged_partner = pd.merge(partner_individual2, partners, how = "right", left_on = "partnership", right_on = "ID")
merged_partner["TRAN_AMT"] = merged_partner["TRAN_AMT"] / merged_partner["length"] # distribute the trade amount evenly to each advisors

firm_info = individuals.iloc[:, :3]
firm_info.drop_duplicates(keep = "first", inplace = True)
merged_partner = pd.merge(merged_partner, firm_info, how = "left", left_on = "individuals", right_on = "ID")

individuals_p = merged_partner[["individuals", "FIRM_NAME_y", "ELITE_FIRM_y", "TRADE_TYPE", "TRAN_ID", "TRAN_DATE", "PROD_ID", "TRAN_AMT"]]
individuals_p.columns = individuals.columns

# now after all the trade data are in individual advisor level, we combine two table together for the further analysis
trades_corrected = pd.concat([individuals, individuals_p]) # vertically concatenate two tables

# Aggregate trade amount
trades_agg = trades_corrected.groupby(["ID"])["TRAN_AMT"].sum()

# Explore relationship between touchpoint and trades
a = touchpoint[touchpoint["ID"] == trades.iloc[0, 0]]
b = trades[trades["ID"] == trades.iloc[0, 0]]
a = a.sort_values(by = "DATE")
b = b.sort_values(by = "TRAN_DATE")

# summary the engagement of each advisor
touchpoint_individual = touchpoint.groupby(["ID"]).agg({'EMAIL_SENT':'sum','EMAIL_OPENED':'sum', 'IN_PERSON_MEETING':'sum',
                                                    'PHONE_ATTEMPT':'sum', 'PHONE_SUCCESS':'sum', 'WEBINAR':'sum',
                                                    'MAIL':'sum'}

# merge touchpoint information onto trades table
all_table = pd.merge(trades_agg, touchpoint_individual, how = "right", left_index = True, right_index = True)
all_table = pd.merge(firm_info, all_table, left_on = "ID", right_index = True)
all_table.set_index("ID", inplace = True)

#%% visualization
