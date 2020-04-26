# -*- coding: utf-8 -*-
"""
@author: Pascal Winter
www.winter-aas.com

Bits and bolts for simple machine learning

1. DATA EXPLORATION
2. FEATURE ENGINEERING
3. GRID SEARCH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

from pathlib import Path
CWD = Path.cwd()




#%%#############################################################################
#BOOK#################### 1. DATA EXPLORATION ##################################
################################################################################
def data_describe(df_data):
    """
    Add further description to describe
    Plot a heatmap with Null values
    """
    # ------------- Calculate a Basic Description  --------------------------------#
    df_des = pd.DataFrame(index=df_data.columns)
    df_des['dtypes'] = df_data.dtypes
    df_des['nunique'] = df_data.nunique()
    df_des['isnull'] = df_data.isnull().sum()
    df_des['null_pc'] = df_des['isnull'] / df_data.shape[0] 
    # link with describe file
    dF_dummy = df_data.describe().transpose()
    df_des = pd.merge(df_des, dF_dummy, how ='left', left_index=True, right_index=True)
    return df_des


def data_classvars(df_des, icatcutoff = 20):
    """
    Propose a variable classification based on a dF describe from above function
    """
    cond_a = df_des['mean'].isnull() == False
    cond_b = df_des['nunique'] < icatcutoff # cutoff at for object vs cat
    df_des.loc[cond_a & ~cond_b, 'Type'] = 'Num'
    df_des.loc[cond_a & cond_b, 'Type'] = 'Cat'
    df_des.loc[~cond_a & ~cond_b, 'Type'] = 'Object'
    df_des.loc[~cond_a & cond_b, 'Type'] = 'Cat'
    # edit the lists
    list_numvar = list(df_des.loc[df_des['Type'] == 'Num'].index)
    list_catvar = list(df_des.loc[df_des['Type'] == 'Cat'].index)
    return list_numvar, list_catvar





#%%#############################################################################
#BOOK#################### 2. FEATURE ENGINEER ##################################
################################################################################

# calculating number of bins with that paper........
# NOT VALIDATED YET
# https://stats.stackexchange.com/questions/197499/what-is-the-best-way-to-decide-bin-size-for-computing-entropy-or-mutual-informat
def calc_bin_size(N):
    ee = np.cbrt(8 + 324*N + 12*np.sqrt(36*N + 729*N**2))
    bins = np.round(ee/6 + 2/(3*ee) + 1/3)
    return int(bins)


def remove_outliers(dF_Data, list_col, dscore):
    """
    Remove the outliers on the rows based on the Z-score threshold defined
    Returns 2 dataframe:
        Datafrae without outliers
        Dataframe containing only outliers
    """
    dF_temp = dF_Data[list_col].apply(scipy.stats.zscore) # Calculate Zscore
    dF_temp = abs(dF_temp) > dscore # Zscore above (or under) threshold
    dF_temp = dF_temp.sum(axis=1) # Sum all to look at overall abnormalities
    dF_Outliers = dF_Data.loc[dF_temp > 0] # Select Outliers
    dF_DataClean = dF_Data.loc[dF_temp == 0] # Remove outliers
    return dF_DataClean, dF_Outliers



#%%#############################################################################
#BOOK######################## 3. GRID SEARCH ###################################
################################################################################


# from sklearn.model_selection import GridSearchCV

# ###############  RANDOM GRid Search for Random Forest 
# parameters = {'max_features': np.arange(5, 10), 'n_estimators':[500], 'min_samples_leaf':[10, 50, 100, 200, 500]}
# random_grid = GridSearchCV(models[1], parameters, cv = 5, verbose=10)
# random_grid.fit(X_train,y_train)
# a = random_grid.cv_results_
# a = random_grid.best_estimator_ 
# a = random_grid.best_params_


# ################  RANDOM GRid Search for Logit 
# parameters = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]} # l1 lasso l2 ridge
# random_grid = GridSearchCV(models[0], parameters, cv=10, verbose=10)
# random_grid.fit(X_train,y_train)
# a = random_grid.best_params_


# ###############  RANDOM GRid Search for GradientBoosting
# parameters = {
#     "loss":["deviance"],
#     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     "min_samples_split": np.linspace(0.1, 0.5, 12),
#     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth":[3,5,8],
#     "max_features":["log2","sqrt"],
#     "criterion": ["friedman_mse",  "mae"],
#     "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators":[10]
#     }
# parameters = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
#               'max_depth': [4, 6, 8],
#               'min_samples_leaf': [20, 50, 100,150],
#               'max_features': [1.0, 0.3, 0.1] 
#               }
# random_grid = GridSearchCV(models[2], parameters, cv=10, verbose=3)
# random_grid.fit(X_train,y_train)
# a = random_grid.best_params_









#%%#############################################################################
#BOOK#################### MODEL ASSESSMENT #####################################
################################################################################

def model_assessment(classifiers, models, X, y, nsplit):
    """
    Assess a series of model with several sim and Provide a Boxplot of dsitrib results
    -------------
    classifiers = ['Logistic Regression','Random Forest','GradientBoosting']
    models = [LogisticRegression(),RandomForestClassifier(n_estimators=100),GradientBoostingClassifier(n_estimators=7,learning_rate=1.1)]
    X: full data predictors
    y: full data target
    nsplit: number of split 
    -----------------
    """
    from sklearn.model_selection import KFold #for K-fold cross validation
    from sklearn.model_selection import cross_val_score #score evaluation
    from sklearn.model_selection import cross_val_predict #prediction
    from sklearn import svm #support vector Machine
    kfold = KFold(n_splits=nsplit, random_state=22) # k=10, split the data into 10 equal parts
    lMean = []
    nAAccuracy = []
    lStdev = [] 
    for i in models:
        model = i
        cv_result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
        lMean.append(cv_result.mean())
        lStdev.append(cv_result.std())
        nAAccuracy.append(cv_result)
    df_Res = pd.DataFrame({'CV Mean':lMean,'Std':lStdev}, index=classifiers)
    df_Acc = pd.DataFrame(nAAccuracy, index=classifiers)
    plt.figure()
    sns.catplot(kind='box', data=df_Acc.T)
    return df_Res, df_Acc







#%%#############################################################################
#BOOK#################### EXOPLANET SPECILA ####################################
################################################################################

def hab_zone(nA_teff, nA_lum, est):
    """
    Caluculate Habitable zone on an nA_Array of Star temperatures (in K)
    http://depts.washington.edu/naivpl/sites/default/files/hz.shtml
    https://arxiv.org/pdf/1404.5292.pdf
    
    nA_teff: star surface temp (K)
    nA_lum: star luminosity (Sun)
    est = 0 : Conservative
    est = 1 : Optimistic
    
    """
    # Calculating HZ fluxes for stars with 2600 K < T_eff < 7200 K. The output file is
    # Coeffcients to be used in the analytical expression to calculate habitable zone flux 
     # i = 0 --> Recent Venus
     # i = 1 --> Runaway Greenhouse (earth)
     # i = 2 --> Maximum Greenhouse (earth)
     # i = 3 --> Early Mars
     # i = 4 --> Runaway Greenhouse for 5 ME
     # i = 5 --> Runaway Greenhouse for 0.1 ME
     # First row: S_effSun(i) 
     # Second row: a(i)
     # Third row:  b(i)
     # Fourth row: c(i)
     # Fifth row:  d(i)
    seffsun  = [1.776, 1.107, 0.356, 0.320, 1.188, 0.99]  # EffectiveSolar Flux
    a = [2.136e-4, 1.332e-4, 6.171e-5, 5.547e-5, 1.433e-4, 1.209e-4]
    b = [2.533e-8, 1.580e-8, 1.698e-9, 1.526e-9, 1.707e-8, 1.404e-8]
    c = [-1.332e-11, -8.308e-12, -3.198e-12, -2.874e-12, -8.968e-12, -7.418e-12]
    d = [-3.097e-15, -1.931e-15, -5.575e-16, -5.011e-16, -2.084e-15, -1.713e-15]
    
    # make sure the temperature is inside the range
    nA_teff = np.minimum(nA_teff, 7200)
    nA_teff = np.maximum(nA_teff, 2600)
    nA_teff = nA_teff - 5780.0
    # ------------------  Calculate Lower Bound 
    #Effective Solar Flux
    if est == 0: i = 1 # Conservative
    if est == 1: i = 0 # Optimistic
    nA_seff_in = seffsun[i] + a[i] * nA_teff + b[i] * nA_teff ** 2 + c[i] * nA_teff ** 3 + d[i] * nA_teff ** 4
    # Derive habitable zone distance
    nA_d_in = np.sqrt(nA_lum / nA_seff_in)
    # ------------------  Calculate Upper Bound 
    #Effective Solar Flux
    if est == 0: i = 2 # Conservative
    if est == 1: i = 3 # Optimistic
    nA_seff_out = seffsun[i] + a[i] * nA_teff + b[i] * nA_teff ** 2 + c[i] * nA_teff ** 3 + d[i] * nA_teff ** 4
    # Derive habitable zone distance
    nA_d_out = np.sqrt(nA_lum / nA_seff_out)   
    return nA_d_in, nA_d_out