# -*- coding: utf-8 -*-
"""
@author: Pascal Winter
www.winter-aas.com
"""



import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import libpw.graphlib as graphlib
import libpw.esglib as esglib



CWD = Path(__file__).resolve().parents[1] # Since we are in a sub-directory



# ------------------------- Results to be loaded ------------------------------#
res_stock_ret =  'run1_ret_db.csv' # for DB format
res_stoc_val =  'run1_val_db.csv' # for DB format
excel_loadfile =  'run1_results.xlsx' # for Matrix Format
csv_loadfile_expret = 'run1_exp_returns.csv' # for the expected returns by funds


i_inputstep_length = 1 # 12: month, 1: year
excel_outputname = 'run1_analysis.xlsx'
rn_sim = True 


load_db_type = True # if yes, load from csv db type. if no load from excel matrix type




# ----------------------------- Graph Parameters ------------------------------#
plt_wd = 345 

#pal = sns.color_palette("Set2")
#pal = sns.color_palette(['black', 'grey', 'white'])
#pal = sns.color_palette('GnBu')
#pal = sns.color_palette('Blues')
pal_name = 'Blues_d'
pal = sns.color_palette(pal_name)
#pal = sns.color_palette('YlGnBu')
sns.set_palette(pal)




#%%#############################################################################
#BOOK############################## LOAD #######################################
################################################################################

# -----------------------------------------------------------------------------#
# --------------------------- Load DB formats  --------------------------------#
# -----------------------------------------------------------------------------#

res_stock_expret_path = list(CWD.rglob(csv_loadfile_expret))[0]
dF_Stock_ExpRet = pd.read_csv(res_stock_expret_path)


if load_db_type == True:
    # ------------------------- Define Load Address
    res_stock_ret_path = list(CWD.rglob(res_stock_ret))[0]
    res_stoc_val_path = list(CWD.rglob(res_stoc_val))[0]
    # ------------------------- Load Data
    # load MP simulations
    dF_Stock_Ret = pd.read_csv(res_stock_ret_path)
    dF_Stock_Val = pd.read_csv(res_stoc_val_path)



# -----------------------------------------------------------------------------#
# --------------------------- Load Matrix formats  ----------------------------#
# -----------------------------------------------------------------------------#
    
if load_db_type == False:
    # get excel file path and prepare map
    xcel_file = pd.ExcelFile(list(CWD.rglob(excel_loadfile))[0])
    map_type = {'ret': 'Return', 'val': 'Price'}
    # predefine DF for output
    dF_Stock_Val = pd.DataFrame(columns=['Year', 'Simulation', 'Price', 'Asset'])
    dF_Stock_Ret = pd.DataFrame(columns=['Year', 'Simulation', 'Return', 'Asset'])
    # go through the sheets
    for sheets in xcel_file.sheet_names:
        # Read from excel
        dF_temp = pd.read_excel(xcel_file, sheets, index_col = 0)
        # Calculate Year
        dF_temp['Year'] = dF_temp.index.values / i_inputstep_length
        # Melt to get a DB format and map the type (ret or val) to get proper column name
        dF_temp = dF_temp.melt(id_vars = 'Year', var_name = 'Simulation', value_name = map_type[sheets[-3:]])
        # Add Asset Name from sheet name
        dF_temp['Asset'] = sheets[:-4] 
        # Append
        if map_type[sheets[-3:]] == 'Return':
            dF_Stock_Ret =  pd.concat([dF_Stock_Ret, dF_temp], axis=0)
        else:
            dF_Stock_Val =  pd.concat([dF_Stock_Val, dF_temp], axis=0)
    # Type correctly simulation
    dF_Stock_Val['Simulation'] = dF_Stock_Val['Simulation'].astype('int64')
    dF_Stock_Ret['Simulation'] = dF_Stock_Ret['Simulation'].astype('int64')






#%%#############################################################################
#BOOK########################### 1.DESCRIBE ####################################
################################################################################

# -----------------------------------------------------------------------------#
# ---------------- Caluclate Mean, Vol and Percentiles  -----------------------#
# -----------------------------------------------------------------------------#

# ------------------- Define Percentiles --------------------------------------#
l_quantile=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]
l_quantile=[0.005, 0.01, 0.25, 0.50, 0.75, 0.9, 0.995]

# ------------------- Calculate Global - RETURNS ------------------------------#
# Percentile returns
dF_Global_StockRet = esglib.calculate_percentile(dF_Stock_Ret, ['Asset'], 'Return', l_quantile)
# Mean returns
dF_temp = esglib.calculate_mean(dF_Stock_Ret, ['Asset'], 'Return', l_quantile)
dF_Global_StockRet = dF_Global_StockRet.append(dF_temp)
# Vol returns
dF_temp = esglib.calculate_vol(dF_Stock_Ret, ['Asset'], 'Return', l_quantile, i_inputstep_length)
dF_Global_StockRet = dF_Global_StockRet.append(dF_temp)


# ------------------- Calculate Year - RETURNS --------------------------------#
# Percentile returns
dF_Period_StockRet = esglib.calculate_percentile(dF_Stock_Ret, ['Asset','Year'], 'Return', l_quantile)
# Mean returns
dF_temp = esglib.calculate_mean(dF_Stock_Ret, ['Asset','Year'], 'Return', l_quantile)
dF_Period_StockRet = dF_Period_StockRet.append(dF_temp)
# Annualize Returns
# dF_Period_StockRet['Return'] = np.power(1 + dF_Period_StockRet['Return'], 12 / i_outpoutstep_length) - 1
# Vol returns
dF_temp = esglib.calculate_vol(dF_Stock_Ret, ['Asset','Year'], 'Return', l_quantile, i_inputstep_length)
dF_Period_StockRet = dF_Period_StockRet.append(dF_temp)



# ------------------- Calculate Year - VALUE ----------------------------------#
# Percentile returns
dF_Period_StockVal = esglib.calculate_percentile(dF_Stock_Val, ['Asset','Year'], 'Price', l_quantile)
# Mean returns
dF_temp = esglib.calculate_mean(dF_Stock_Val, ['Asset','Year'], 'Price', l_quantile)
dF_Period_StockVal = dF_Period_StockVal.append(dF_temp)







#%%#############################################################################
#BOOK############################## 2.GRAPH ####################################
################################################################################

# -----------------------------------------------------------------------------#
# -------------------- 2.1 Graph Mean Returns  --------------------------------#
# -----------------------------------------------------------------------------#

# ----------------------------- Settings --------------------------------------#
lassets = list(dF_Stock_Ret['Asset'].unique())
nassets = len(lassets)
fig, axes = plt.subplots(nassets, 1, figsize=graphlib.set_size(plt_wd, nassets, 1),
                         sharex=True)

# Select Mean by Asset and Year
dF_temp = dF_Period_StockRet.reset_index()
cond_2 = dF_temp['Indicator'] == 'Mean'
dF_temp2 = dF_temp.loc[cond_2] # Select mean

for i, asset in enumerate(lassets):
    dF_temp4 = dF_temp2.loc[dF_temp2['Asset'] == asset]
    dF_temp3 = dF_Stock_ExpRet.loc[dF_Stock_ExpRet['StockName'] == asset]
    sns.lineplot(data=dF_temp4, x='Year', y='Return', ax=axes[i], color=pal[0])
    sns.lineplot(data=dF_temp3, x='Year', y='ExpRet', ax=axes[i], color=pal[5])

# ----------------------------- Graph Cosmetics -------------------------------#
# Remove the border
sns.despine()
# Adjust space inbetween columns
fig.subplots_adjust(wspace = 0.3)

# Add the Y Axis titles
for i, asset in enumerate(lassets):
    axes[i].set_ylabel(asset)
    # Set percentage for returns
    axes[i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))





#%% ---------------------------------------------------------------------------#
# -------------------- 2.2 Graph Percentiles and Mean  ------------------------#
# -----------------------------------------------------------------------------#

# ----------------------------- Settings --------------------------------------#
lassets = list(dF_Stock_Ret['Asset'].unique())
nassets = len(lassets)
pal_temp = sns.diverging_palette(240, 240, n=len(l_quantile))
fig, axes = plt.subplots(nassets, 3, figsize=graphlib.set_size(plt_wd, nassets, 3),
                         sharex=True, sharey='col')

# ---------------- 1st  Column: Price overview ----------
# Calculate quantiles by Asset and Year
dF_temp = dF_Period_StockVal.reset_index()
cond_1 = dF_temp['Indicator'] != 'Mean'
cond_2 = dF_temp['Indicator'] == 'Mean'
dF_temp1 = dF_temp.loc[cond_1] # Select percentile
dF_temp2 = dF_temp.loc[cond_2] # Select mean
# Graphing
for i, asset in enumerate(lassets):
    dF_temp3 = dF_temp1.loc[dF_temp['Asset'] == asset]
    dF_temp4 = dF_temp2.loc[dF_temp2['Asset'] == asset]
    sns.lineplot(data=dF_temp3, x='Year', y='Price',hue='Indicator', ax=axes[i,0],
              palette=pal_temp, legend=False)
    sns.lineplot(data=dF_temp4, x='Year', y='Price', ax=axes[i,0], color=pal[0])
    # Styling curves
    #for j, curves in enumerate(l_quantile):
    #    axes[i,0].lines[j].set_linestyle("--")

# ---------------- 2nd  Column: Return overview ----------
# Calculate quantiles by Asset and Year
dF_temp = dF_Period_StockRet.reset_index()
cond_1 = dF_temp['Indicator'] != 'Mean'
cond_1b = dF_temp['Indicator'] != 'Vol'
cond_2 = dF_temp['Indicator'] == 'Mean'
dF_temp1 = dF_temp.loc[cond_1 & cond_1b] # Select percentile
dF_temp2 = dF_temp.loc[cond_2] # Select mean
# Graphing
for i, asset in enumerate(lassets):
    dF_temp3 = dF_temp1.loc[dF_temp1['Asset'] == asset]
    dF_temp4 = dF_temp2.loc[dF_temp2['Asset'] == asset]
    sns.lineplot(data=dF_temp3, x='Year', y='Return',hue='Indicator', ax=axes[i,1],
              palette=pal_temp, legend=False)
    sns.lineplot(data=dF_temp4, x='Year', y='Return', ax=axes[i,1], color=pal[0])
    # Styling curves
    #for j, curves in enumerate(l_quantile):
    #axes[i,1].lines[j].set_linestyle("--")



# ---------------- 3rd  Column: Volatility ----------
# Calculate quantiles by Asset and Year
dF_temp = dF_Period_StockRet.reset_index()
cond_1 = dF_temp['Indicator'] == 'Vol'
dF_temp1 = dF_temp.loc[cond_1] # Select vol
# Graphing
for i, asset in enumerate(lassets):
    dF_temp2 = dF_temp1.loc[dF_temp['Asset'] == asset]
    sns.lineplot(data=dF_temp2, x='Year', y='Return', ax=axes[i,2], color=pal[0])



# ----------------------------- Graph Cosmetics -------------------------------#
# Remove the border
sns.despine()
# Adjust space inbetween columns
fig.subplots_adjust(wspace = 0.3)

# Add the Y Axis titles
for i, asset in enumerate(lassets):
    axes[i,0].set_ylabel(asset)
    axes[i,1].set_ylabel("")
    axes[i,2].set_ylabel("")
    # Set logscale for Values
    axes[i,0].set_yscale('log')
    # Set percentage for returns
    axes[i,1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    axes[i,2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# axes[nassets-1,2].legend(labels=l_quantile)     



#%%#############################################################################
############################## EXPORT  #########################################
################################################################################


#--------------------------  Output the results -------------------------------#
spath = res_stock_ret_path.parent.as_posix() + "/" + excel_outputname


print('Exporting Results to Excel...')
# Setup excel writer
writer = pd.ExcelWriter(spath, engine='xlsxwriter') 
# Write Files - Single Simulation
dF_Global_StockRet.to_excel(writer, sheet_name='Global_Stock', freeze_panes=(1,1))
dF_Period_StockVal.unstack(level="Year").to_excel(writer, sheet_name='Year_Stock_Val', freeze_panes=(1,1))
dF_Period_StockRet.unstack(level="Year").to_excel(writer, sheet_name='Year_Stock_Ret', freeze_panes=(1,1))
# Close writer
writer.save()

