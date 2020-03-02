# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:42:05 2020

@author: XiaoPi
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import libpw.dalib as dalib
import libpw.graphlib as graphlib

from pathlib import Path
CWD = Path('__file__').parent

# Set the color palette
#plt_wd = graphlib.plt_wd _palette("Paired"))
plt_wd = graphlib.plt_wd # set plot width


""" NOTES
Good for:
    - Classification algorithm
    - Graphing
    - Exoplanet is cool stuff ;)


"""



#%%#############################################################################
#BOOK######################### INITIALISATION ##################################
################################################################################
# https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html
# https://exoplanetarchive.ipac.caltech.edu/docs/API_compositepars_columns.html

# ----------------------------- Import Dataset --------------------------------#
# csv_kepler = list(CWD.rglob('kepler.csv'))[0]
# dF_Data = pd.read_csv(csv_kepler)
csv_exo = list(CWD.rglob('compositepars_2020.02.16_08.15.02.csv'))[0]
dF_Data = pd.read_csv(csv_exo, skiprows = 109)



# ------------- Calculate Planet distance to its star -------------------------#
# Calculate planet's Star Standard Gravitional Parameter (AU3 * SolarMass-1 * year-2) * SolarMass = AU3 * year-2
dF_temp = 39.478 * dF_Data['fst_mass'] 
# Multiply by orbital period square in years and divide by 4pi square (AU3)
dF_temp = dF_temp *  np.power(dF_Data['fpl_orbper'] / 365.25 , 2) / (4 * np.power(np.pi, 2))
# Take the cubic root to get orbit semi major axis (AU)
dF_Data['fpl_diststar']  = np.power(dF_temp, 1/3)

# Test
# cond_1 = abs(dF_Data['fpl_orbper'] -365) < 20
# cond_2 = abs(dF_Data['fst_mass'] -1) < 0.2
# a = dF_Data.loc[cond_1 & cond_2]

# ------------- Calculate Star Habitable Zone  Approximation ------------------#
dF_Data['fst_hzin'], dF_Data['fst_hzout'] = dalib.hab_zone(dF_Data['fst_teff'], np.exp(dF_Data['fst_lum']), 1)
# Define zone
dF_Data['fpl_hzone'] = (dF_Data['fpl_diststar'] >= dF_Data['fst_hzin'])*1 + (dF_Data['fpl_diststar'] >= dF_Data['fst_hzout'])*1


# ----------------------- Test density ----------------------------------------#
# a1 = dF_Data2['fpl_bmasse'] * 5.972 * np.power(10.0, 12)  # Change mass in 10^12 kg
# a2 = dF_Data2['fpl_rade'] * 6371.0 # Change radius in km
# a3 = 4 / 3 * np.pi * np.power(a2, 3) # Calculate volume in km3
# a4 = a1 / a3 # Calulcate density in 10^12 kg / km3

# # Earh mass: 5.972 × 10^24 kg
# # Earth radius: 6,371 km
# # g/ cm3 ~ 10^12 kg / km3


#%%#############################################################################
#BOOK########################## SELECT DATA ####################################
################################################################################

# --------------------------- General -----------------------------------------#
list_m1 = [
         'fpl_letter',      # 
         'fpl_discmethod',  # Discovery Method 	
         ]
# --------------------------- Planet  -----------------------------------------#
list_m2 = [
         'fpl_orbper',      # Orbital Period (days)
         'fpl_bmasse',      # Mass (earth)
         'fpl_rade',        # Radius (earth)
         'fpl_dens',        # Planet Density (g/cm**3)
         'fpl_diststar',    # Distance to its star (AU)
         'fpl_hzone',       # Is in habitable Zone ?: 0: Hot, 1: Yes, 2: Cold                  
         ]
# -------------------------- Stellar  -----------------------------------------#
list_m3 = [  
          ]

list_m_all = list_m1 + list_m2 + list_m3

# -------------------------- Prepare the data ---------------------------------#
dF_Data2 = dF_Data[list_m_all].copy() # Select columns
dF_Data2 = dF_Data2.dropna()  # Drop NA

# --------------------------- Remove Outliers ---------------------------------#
list_temp = ['fpl_orbper', 'fpl_bmasse', 'fpl_rade', 'fpl_dens']#, 'fst_rad', 'fst_mass', 'fst_teff']
dF_Data2, dF_Outliers = dalib.remove_outliers(dF_Data2, list_temp, 3)
dF_Data2, dF_Outliers2 = dalib.remove_outliers(dF_Data2, list_temp, 10)




#%%#############################################################################
#BOOK######################## DESCRIBE DATA ####################################
################################################################################


#----------------------------- Describe the data ------------------------------#
dF_des = dalib.data_describe(dF_Data2)
list_numvar, list_catvar = dalib.data_classvars(dF_des, 30)



#------------------------- Produce a Is Na heatmap ----------------------------#
fig, axs = plt.subplots(2, 1, figsize=graphlib.set_size(plt_wd, 2, 1))
sns.heatmap(dF_Data.isnull(), cbar=False, cmap='viridis', ax = axs[0])
sns.heatmap(dF_Data2.isnull(), cbar=False, cmap='viridis', ax = axs[1])


# #-------------------------------- Graph the data ------------------------------#
# #--------------- Univariate
graphlib.graph_univar(dF_Data2, list_numvar, 'num')
graphlib.graph_univar(dF_Data2, list_catvar, 'cat')
# #--------------- Univariate by cat (simple and details)
# graphlib.graph_univar_pred(dF_Data2, list_numvar, 'fpl_discmethod', 'num')
# graphlib.graph_univar_pred(dF_Data2, list_catvar, 'fpl_discmethod', 'cat')
# graphlib.graph_univar_pred2(dF_Data2, list_numvar, 'fpl_discmethod', 'num')
# graphlib.graph_univar_pred2(dF_Data2, list_catvar, 'fpl_discmethod', 'cat')
# ## --------------- Bivariate Numcat
# graphlib.graph_multvar_numcat(dF_Data2, list_numvar, list_catvar)
# ## --------------- Bivariate Numnum
# graphlib.graph_multvar_numnum(dF_Data2, list_numvar)
# graphlib.graph_multvar_numnum(dF_Data2, list_numvar, hue = 'fpl_discmethod')
# ## --------------- Bivariate Catcat
# graphlib.graph_multvar_catcat(dF_Data2, list_catvar)


#-------------------------- Look at correlations ------------------------------#
# Look at all correlations (big figure)
fig, axs = plt.subplots(1, 1, figsize=graphlib.set_size(plt_wd, 1, 1))
sns.heatmap(dF_Data2.corr(), annot=True, linewidths=0.2, ax = axs,
            vmin=-0.5, vmax=0.5, center= 0, cmap='coolwarm')





#%%#############################################################################
#BOOK###################### FEATURE ENGINEERING ################################
################################################################################

dF_Data3 = dF_Data2.copy()


# ----------------------- Mapping Values & Numerize ---------------------------#
# Planet Assigned Letter 
dict_map ={'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
dF_Data3['fpl_letter'] = dF_Data3 ['fpl_letter'].replace(dict_map)



# ----------------------- Mapping Values & Dummify ----------------------------#
# ----------- fpl_discmethod
dict_map ={'Radial Velocity': 'RV',
           'Transit': 'Tr',
           'Eclipse Timing Variations' : 'Oth',
           'Pulsation Timing Variations': 'Oth',
           'Transit Timing Variations': 'Oth',
           'Orbital Brightness Modulation': 'Oth'}
dF_Data3['fpl_discmethod'] = dF_Data3 ['fpl_discmethod'].replace(dict_map)
# Hot encoding
dF_dummy = pd.get_dummies(dF_Data3['fpl_discmethod'], prefix = 'fpl_discm_' )
dF_Data3 = pd.concat([dF_Data3, dF_dummy], axis = 1)
dF_Data3 = dF_Data3.drop('fpl_discmethod', axis=1)
dF_Data3 = dF_Data3.drop('fpl_discm__Oth', axis=1)



# --------------------------- Lognormalize ------------------------------------#
list_temp = ['fpl_orbper', 'fpl_bmasse', 'fpl_rade', 'fpl_dens' , 'fpl_diststar']#, 'fst_dist', 'fst_mass', 'fst_rad']
for col in list_temp:
    dF_Data3[col] = (dF_Data3[col] -  dF_Data3[col].mean()) / dF_Data3[col].std()
    dF_Data3[col] =  dF_Data3[col] - dF_Data3[col].min() + 1
    dF_Data3[col] = np.log(dF_Data3[col])
    

# ------------------------------ Normalize ------------------------------------#
# list_temp = ['fst_optmag', 'fst_nirmag', 'fst_teff', 'fst_logg', 'fst_lum', 'fst_met']
# for col in list_temp:
#     dF_Data3[col] = (dF_Data3[col] -  dF_Data3[col].mean()) / dF_Data3[col].std()






#%%#############################################################################
#BOOK###################### MACHINE LEARNING ###################################
################################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
#from sklearn.cluster import MeanShift, estimate_bandwidth



# --------------------- Select and Normalise the dataset ----------------------#
dF_DataClust = dF_Data3[list_m2]
X = StandardScaler().fit_transform(dF_Data3[list_m2])




# -------------------- Initialise the learning --------------------------------#
# Model Names (the size will define the # of models)
lModelName = ['Kmeans','MeanShift','Spectral', 'Agglom', 'DBScan']

# ---------------- Define Model parameters
param = [{} for i in range(len(lModelName))]
# Parameters for Kmeans
param[0] = {'n_clusters': 4}
# Parameters for MeanShift
param[1] = {}
# Parameter for Spectral
param[2] = {'n_clusters': 4,
            'assign_labels': "discretize"}
# Parameter for Agglom
param[3] = {'n_clusters': 4}
# Parameter for DBSCAN
param[4] = {'eps': 0.76}


# ------------------ Define Models
models = [KMeans(**param[0]),
          MeanShift(**param[1]),
          SpectralClustering(**param[2]),
          AgglomerativeClustering(**param[3]),
          DBSCAN(**param[4])]




# Best so far: 0, 3
#%%--------------------------- Run the learning -------------------------------#
i = 4 # Select the model 
km = models[i].fit(X)
dF_Data2['Labels'] = km.labels_
dF_Data2['Labels'] ="Cat_" + dF_Data2['Labels'].astype(str)




#%%-------------------------- Graph Results -----------------------------------#
#graphlib.graph_univar_pred2(dF_Data2, list_numvar, 'Labels', 'num')
#graphlib.graph_univar_pred2(dF_Data2, list_catvar, 'Labels', 'cat')

graphlib.graph_multvar_numnum(dF_Data2, list_numvar, hue ='Labels')
#graphlib.graph_multvar_catcat(dF_Data2, list_catvar)




# ------------------------------ Old stuff ------------------------------------#

# dF_DataClust = dF_Data3[list_m2]
# list_clust = []
# for i in range(1, 11):
#     km = KMeans(n_clusters=i).fit(dF_DataClust)
#     list_clust.append(km.inertia_)
    
# fig, ax = plt.subplots(figsize=graphlib.set_size(plt_wd, 1, 1))
# sns.lineplot(x=list(range(1, 11)), y=list_clust, ax=ax)

# km = KMeans(n_clusters=3).fit(dF_DataClust)
# dF_Data2['Labels'] = km.labels_
# dF_Data2['Labels'] ="Cat_" + dF_Data2['Labels'].astype(str)





# graphlib.graph_univar_pred(dF_Data2, list_numvar, 'Labels', 'num')
# graphlib.graph_univar_pred(dF_Data2, list_catvar, 'Labels', 'cat')

# graphlib.graph_univar_pred2(dF_Data2, list_numvar, 'Labels', 'num')
# graphlib.graph_univar_pred2(dF_Data2, list_catvar, 'Labels', 'cat')

# graphlib.graph_multvar_numnum(dF_Data2, list_numvar, hue ='Labels')
# graphlib.graph_multvar_catcat(dF_Data2, list_catvar)







