# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:42:05 2020

@author: XiaoPi
"""

""" NOTES
----------- Variable Selection -------------
=> can try a run with [fpl_insol + fst_age]

----------- Feat Eng -------------
=> Distance from the star and Habitable zone
=> Heat estimnation (see above)
=> Nirmag vs OPmag
=> Star classification first 


"""





#%%#############################################################################
#BOOK############################ FIELD DEFIN ##################################
################################################################################
# --------------------------- General -----------------------------------------#
list_a1 = ['fpl_hostname',    # 
         'fpl_letter',      # 
         'fpl_name',        # 
         'fpl_discmethod',  # Discovery Method 	
         'fpl_controvflag', # 
         'fpl_disc']        # Discovery Year

# --------------------------- Planet  -----------------------------------------#
list_a2 = ['fpl_orbper',      # Orbital Period (days)
         'fpl_smax',        # Orbital major semi axis (au)
         'fpl_eccen',       # Eccentricity
         'fpl_bmasse',      # Mass (earth)
         'fpl_bmassprov',   # Mass provenance
         'fpl_rade',        # Radius (earth)
         'fpl_dens',        # Planet Density (g/cm**3)
         'fpl_eqt',         # Planet Equilibrium Temperature [K]
         'fpl_insol',       # Insolation Flux [Earth flux]
         'fpl_diststar',    # Distance to its star (AU)
         'fpl_hzone',       # Is in habitable Zone ?: 0: Hot, 1: Yes, 2: Cold
         ]

# -------------------------- Stellar  -----------------------------------------#
list_a3 = ['ra',              # Right Ascension (dec degree)
         'dec',             # Declination (dec degree)
         'fst_dist',        # Dist (parsec)
         'fst_optmag',      # Optical Magnitude [mag]
         'fst_optmagband',  # Optmag Band
         'fst_nirmag',      # Near-IR Magnitude [mag]
         'fst_nirmagband',  # Near-IR Magnitude Band
         'fst_spt',         # Spectral Type
         'fst_teff',        # Effective Temperature [K]
         'fst_logg',        # Stellar Surface Gravity [log10(cm/s**2)]
         'fst_lum',         # Stellar Luminosity [log(Solar luminosity)]
         'fst_mass',        # Stellar Mass [Solar mass]
         'fst_rad',         # Stellar Radius [Solar radii]
         'fst_met',         # Stellar Metallicity [dex]
         'fst_metratio',    # Metallicity Ratio
         'fst_age',         # Stellar Age [Gyr]
         'fst_hzin',        # Habitable Zone inner [AU] 
         'fst_hzout',       # Habitable Zone outer [AU]          
         ]         

list_a_all = list_a1 + list_a2 + list_a3
dF_Data2 = dF_Data[list_a_all]




#%%#############################################################################
#BOOK########################### FIELD DEFIN2 ##################################
################################################################################

# Retain only parameters with limited NaN


# --------------------------- General -----------------------------------------#
list_b1 = [
         'fpl_hostname',    # 
         'fpl_letter',      # 
         'fpl_name',        # 
         'fpl_discmethod',  # Discovery Method 	
         'fpl_controvflag', # 
         'fpl_disc']        # Discovery Year

# --------------------------- Planet  -----------------------------------------#
list_b2 = [
         'fpl_orbper',      # Orbital Period (days)
         'fpl_bmasse',      # Mass (earth)
         'fpl_rade',        # Radius (earth)
         'fpl_dens',        # Planet Density (g/cm**3)
         #'fpl_eqt',        # Planet Equilibrium Temperature [K]    / IMPORTANT but lots of NAN
         #'fpl_insol'       # Insolation Flux [Earth flux]          / IMPORTANT but half of NAN
         'fpl_diststar',    # Distance to its star (AU)
         'fpl_hzone',       # Is in habitable Zone ?: 0: Hot, 1: Yes, 2: Cold
         ]

# -------------------------- Stellar  -----------------------------------------#
list_b3 = [
         'fst_dist',        # Dist (parsec)
         'fst_optmag',      # Optical Magnitude [mag]
         'fst_nirmag',      # Near-IR Magnitude [mag]
         #'fst_spt',         # Spectral Type                        / IMPORTANT but lots of NAN
         'fst_teff',        # Effective Temperature [K]
         'fst_logg',        # Stellar Surface Gravity [log10(cm/s**2)]
         'fst_lum',         # Stellar Luminosity [log(Solar luminosity)]
         'fst_mass',        # Stellar Mass [Solar mass]
         'fst_rad',         # Stellar Radius [Solar radii]
         'fst_met',         # Stellar Metallicity [dex]
         'fst_metratio',    # Metallicity Ratio
         #'fst_age']         # Stellar Age [Gyr]                    / IMPORTANT but quarter of NAN
         'fst_hzin',        # Habitable Zone inner [AU] 
         'fst_hzout',        # Habitable Zone outer [AU]   
         ]

list_b_all = list_b1 + list_b2 + list_b3


# -------------------------- Prepare the data ---------------------------------#
dF_Data2 = dF_Data[list_b_all] # Select columns
dF_Data2 = dF_Data2.dropna()  # Drop NA


# --------------------------- Remove Outliers ---------------------------------#
list_temp = ['fpl_orbper', 'fpl_bmasse','fpl_rade', 'fpl_dens', 'fst_rad', 'fst_mass', 'fst_teff']
dF_Data2, dF_Outliers = dalib.remove_outliers(dF_Data2, list_temp, 3)
dF_Data2, dF_Outliers2 = dalib.remove_outliers(dF_Data2, list_temp, 10)

