
# -*- coding: utf-8 -*-
"""
@author: Pascal Winter
www.winter-aas.com
"""

import pandas as pd
import numpy as np

import libpw.esglib as esglib

from pathlib import Path
CWD = Path(__file__).parent

'''
--------------------------- Further work to be done ---------------------------
Interest model
Optimisation on append + Numpyfication for speed


----------------------------------- Content -----------------------------------
0. PARAMETERS / INITITIALISATION
1. RANDOM NUMBER GENERATION
2. STOCK MODEL - B&S
3. INTEREST RATE MODEL - CIR 

----------------------------------- Inputs ------------------------------------
Correlation starts with stocks then rates



----------------------------------- Outputs -----------------------------------
Stock Model: Returns 2 csv file in a DB format with fields ['Asset', 'Simulation', 'Year', X]
    => X:Values - starts from output step 0 - value starting at 100 at bop 
    => X:Returns - starts from output step 1

'''
# -----------------------------------------------------------------------------#
# ------------------------------- Options -------------------------------------#
# -----------------------------------------------------------------------------#

# --------------------- Simulation Parameters ---------------------------------#
i_num_sim = 2000 #10000
i_num_steps = 55 # projection year
i_step_length = 48 # Step Per years (in calculation)
d_deltaT = 1 / i_step_length
i_outpoutstep_length = 1 # 12: month, 1: year

# --------------------- Random Generation -------------------------------------#
seed_rand = True
seed_val = 32435

# --------------------- Technical ---------------------------------------------#
rn_sim = True # if True, asset return will be the yield curve minus div yield

# ---------------------- Files Path -------------------------------------------#
# Input
excel_parameters = 'run1_parameters.xlsx' 
# Outputs
csv_output = 'run1_val_db.csv'
csv_output2 = 'run1_ret_db.csv'
xlsx_outputname = 'run1_results.xlsx'
csv_eret_outputname = 'run1_exp_returns.csv'

output_db = True
output_xlsx = True 



#%%#############################################################################
########################## 0. PARAMETERS / INITI ###############################
################################################################################
 

# --------------------- Load tables Parameters --------------------------------#
xcel_file = pd.ExcelFile(list(CWD.rglob(excel_parameters))[0])
dF_StockParam = pd.read_excel(xcel_file, 'Stock_Param', index_col = 0)
dF_IntParam = pd.read_excel(xcel_file, 'Int_Param', index_col = 0)
dF_YieldCurve = pd.read_excel(xcel_file, 'Yield_Curve', index_col = 0)
nA_Correlation = pd.read_excel(xcel_file, 'Correlation', header = None)
nA_Correlation = nA_Correlation.to_numpy()


# Check consistency
dF_Test = pd.DataFrame(columns = ['Name','Val1','Val2'])
i_num_assets = dF_StockParam.shape[0] + dF_IntParam.shape[0]
dF_Test.loc[dF_Test.shape[0]] = ['Import - Correl Matrix Size', nA_Correlation.shape[0],
            dF_StockParam.shape[0] + dF_IntParam.shape[0]]


# --------------------------- Global variables --------------------------------#
# Define Indexes to be used throughout the simulation
l_asset = list(dF_StockParam.index) + list(dF_IntParam.index)
l_stocks = list(dF_StockParam.index)
l_simulation = list(np.arange(0, i_num_sim))
# Define time steps to be used
i_step_modulo = i_step_length // i_outpoutstep_length # modulo for extraction output
l_step_out = list(np.arange(0, i_step_length * i_num_steps + 1, i_step_modulo)) # selected steps for extraction
# Steps for indexes
l_step_year = list(np.arange(0, i_step_length * i_num_steps, i_step_modulo) * d_deltaT ) # selected steps for extraction
l_step_year2 = list(np.arange(0, i_step_length * i_num_steps + 1, i_step_modulo) * d_deltaT ) # selected steps for extraction


# Define Stock Indexes
Stock_index = pd.MultiIndex.from_product([l_simulation, l_step_year, l_stocks],
                                         names=[ 'Simulation', 'Year', 'Asset'])
Stock_index2 = pd.MultiIndex.from_product([l_simulation, l_step_year2, l_stocks],
                                         names=[ 'Simulation', 'Year', 'Asset'])


# --------------------- Align Yield Curve on time frame -----------------------#
# with interpolation
tindex = np.linspace(0,  dF_YieldCurve.index.max(), int(i_step_length * dF_YieldCurve.index.max() + 1))
dF_YC_Aligned =  dF_YieldCurve.reindex(tindex)
dF_YC_Aligned = dF_YC_Aligned.interpolate(method = 'linear' )
dF_YC_Aligned = dF_YC_Aligned[:i_num_steps].iloc[:-1] # trim to get same size
dF_YC_Aligned = dF_YC_Aligned.reset_index()






#%%#############################################################################
####################### 1. RANDOM NUMBER GENERATION  ###########################
################################################################################
# Creates a nA of size Sim * Time * Asset

# -----------------------------------------------------------------------------#
# ----------------Create the random numbers with a normal distribution --------#
# -----------------------------------------------------------------------------#

# Initialise mean and covariance
mean = [0] * i_num_assets 
cov = nA_Correlation
# Seed the random number generator
if seed_rand == True: np.random.seed(seed_val)
# Generate the Random multivariate vector (size: ( Time * Sim) * Asset)
nA_Multvar = np.random.multivariate_normal(mean, cov, 
                        size=(i_step_length * i_num_steps  *i_num_sim))
# Reshape as an Sim * Time * Asset array
nA_Multvar = nA_Multvar.reshape(i_num_sim, i_step_length * i_num_steps, i_num_assets)





#%%#############################################################################
############################# 2. STOCK MODEL - B&S  ############################
################################################################################

# -----------------------------------------------------------------------------#
# ------------------ Initialise for the stock calculation  --------------------#
# -----------------------------------------------------------------------------#

l_stocks = list(dF_StockParam.index) 
n_stocks = len(l_stocks)
nA_Multvar2 = nA_Multvar[:, :, 0:n_stocks] # Slice the Multvar for further calculation

nA_StockBS = np.zeros( shape = nA_Multvar2.shape, dtype = 'float32')


# -----------------------------------------------------------------------------#
# ------------------ Calculate the B/S returns --------------------------------#
# -----------------------------------------------------------------------------#

# ------------------------ Calculate the 1st BS Term --------------------------#
# Apply the forward curves if this is RN, assumed returns if RW
if rn_sim == True: 
    nA_StockBS = np.add(nA_StockBS, dF_YC_Aligned['Forward'][None, :, None])
else:
    nA_StockBS = np.add(nA_StockBS, dF_StockParam['Return'][None, None, :])
# Apply the dividends
nA_StockBS = np.subtract(nA_StockBS, dF_StockParam['Dividend'][None, None, :])
# Lognormalise
nA_StockBS = np.log(1 + nA_StockBS)
# Substract the volatility term
nA_temp = np.square(dF_StockParam['Volatility']) / 2
nA_StockBS = np.subtract(nA_StockBS, nA_temp[None, None, :])
# Multiply by DeltaT
nA_StockBS = nA_StockBS * d_deltaT

# ------------------------ Calculate the 2nd BS Term --------------------------#
# Multiply the multivariate term with the vol
nA_Multvar2 = np.multiply(nA_Multvar2, dF_StockParam['Volatility'][None, None, :])
# Scale by Sqrt of DeltaT
nA_Multvar2 = nA_Multvar2 * np.sqrt(d_deltaT)
# Add 1st and 2nd term
nA_StockBS = nA_StockBS + nA_Multvar2
# Exponentialise
nA_StockBS = np.exp(nA_StockBS)



# -----------------------------------------------------------------------------#
# ------------------------ Wrangle the data for output ------------------------#
# -----------------------------------------------------------------------------#

# -------- Calculate Value and extract Value and Returns at output step  ------#
# Insert first value and add 1 to all returns
nA_StockBS_Val = np.insert(nA_StockBS, 0, 1.0, axis = 1)
# Cumulative Product to get Value
nA_StockBS_Val = nA_StockBS_Val.cumprod(axis=1)
# Select only the output steps
nA_StockBS_Val_Out = nA_StockBS_Val[:, l_step_out, :]
# Calculate the Output Steps returns
nA_StockBS_Ret_Out = nA_StockBS_Val_Out[:, 1:, :] / nA_StockBS_Val_Out[:, :-1, :] - 1


# ------------------------- Reshape on a DB format for output -----------------#
# flatten numpy as 1 dimension and pass it as a dataframe
dF_StockBS_Val_Out = pd.DataFrame(nA_StockBS_Val_Out.flatten(), 
                                  index = Stock_index2, columns = ['Price'])


dF_StockBS_Ret_Out = pd.DataFrame(nA_StockBS_Ret_Out.flatten(), 
                                  index = Stock_index, columns = ['Return'])






#%%#############################################################################
########################## 2. INTEREST RATE MODEL ##############################
################################################################################

# TO BE DEVELOPPED







#%%#############################################################################
################################  TEST  ########################################
################################################################################

#----------------------- Print the test dF ------------------------------------#
dF_Test['Diff'] = dF_Test['Val1'] - dF_Test['Val2']
dF_Test['Result'] = abs(dF_Test['Diff']) > 0.00000001
print(dF_Test)




################################################################################
######################### EXPORT & TEST ########################################
################################################################################



# -----------------------------------------------------------------------------#
#---------------------  Export results - DB -----------------------------------#
# -----------------------------------------------------------------------------#


if output_db == True:
    # Values
    print('Exporting DB to csv...')
    spath = list(CWD.rglob(excel_parameters))[0].parent.as_posix() + "/" + csv_output
    dF_StockBS_Val_Out.to_csv(spath)
    # Returns
    spath = list(CWD.rglob(excel_parameters))[0].parent.as_posix() + "/" + csv_output2
    dF_StockBS_Ret_Out.to_csv(spath)


# -----------------------------------------------------------------------------#
#---------------------  Export results - Matrix -------------------------------#
# -----------------------------------------------------------------------------#
# Format Time * Simulation

if output_xlsx == True:
    print('Exporting Results to Excel...')
    # Setup excel writer
    spath = list(CWD.rglob(excel_parameters))[0].parent.as_posix() + "/" + xlsx_outputname
    writer = pd.ExcelWriter(spath, engine='xlsxwriter') 
    # Write Files - Single Simulation
    for i, stock in enumerate(l_stocks):
        dF_temp = pd.DataFrame(nA_StockBS_Val_Out[:, :, i])
        name_sheet = stock + '_val'
        dF_temp.T.to_excel(writer, sheet_name=name_sheet, freeze_panes=(1,1))
        dF_temp = pd.DataFrame(nA_StockBS_Ret_Out[:, :, i])
        name_sheet = stock + '_ret'
        dF_temp.T.to_excel(writer, sheet_name=name_sheet, freeze_panes=(1,1))    
    # Close writer
    writer.save()
    writer.close()



# -----------------------------------------------------------------------------#
#---------------------  Export Expected Returns -------------------------------#
# -----------------------------------------------------------------------------#

spath = list(CWD.rglob(excel_parameters))[0].parent.as_posix() + "/" + csv_eret_outputname
# ------------------------ Calculate the expected returns
dF_temp = dF_YC_Aligned * rn_sim # add YC only for Risk neutral cases
# Develop the dataframe with the stock indexes
dF_temp = dF_temp.assign(key=1).merge(dF_StockParam.reset_index().assign(key=1), on='key').drop('key', 1)
# Develop the dataframe with the stock indexes
dF_temp['ExpRet'] = dF_temp['Return'] * (1 -  rn_sim ) # add Returns only for RW cases
dF_temp['ExpRet'] = dF_temp['ExpRet'] + dF_temp['Forward'] - dF_temp['Dividend']
dF_temp = dF_temp[['Year', 'StockName', 'ExpRet']]
dF_temp.to_csv(spath)



