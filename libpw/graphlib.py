# -*- coding: utf-8 -*-
"""
@author: Pascal Winter
www.winter-aas.com


Bits and bolts for visual data exploration

1. DATA EXPLORATION
2. UNIVARIATE GRAPHING
3. MULTIVARIATE GRAPHING
4. VRAC - Bit and Bolt




"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path
CWD = Path.cwd()

plt_wd = 345 * 2 # In points






#%%#############################################################################
#BOOK#################### 1. DATA EXPLOTRATION #################################
################################################################################

#%%------------------------------ Set Size ------------------------------------#
def set_size(width, norow, nocol, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.
    -------- Parameters:
        width: float / Width in pts (345)
        norow: int / # row in subblot
        fraction: float / Fraction of the width which you wish the figure to occupy
    -------- Returns:
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in * nocol, fig_height_in * norow)
    return fig_dim



#%%#############################################################################
#BOOK#################### 2. UNIVARIATE GRAPHING ###############################
################################################################################


def graph_univar(df_data, list_var, var_type):
    """
    Graph very simple univariate for each variable
    var_type = 
        "num" if numeric:       Distplot
        "cat" if categorical:   Countplot
    """
    #----------- Histograms for Numerical variable
    # Setup the graph subplot
    fig, axs = plt.subplots(len(list_var), 1, 
            figsize=set_size(plt_wd, len(list_var), 1))
    fig.subplots_adjust(hspace = 0.3) # Adjust space between rows
    # loop for graphing
    for i, item in enumerate(list_var):
        cond_a = df_data[item].isna() == False
        dF_dummy = df_data.loc[cond_a]
        if var_type == "num":
            sns.distplot(dF_dummy[item], ax = axs[i])
        if var_type == "cat":
            sns.countplot(dF_dummy[item], ax = axs[i])
            # Add the percentage in the graph
            total = dF_dummy.shape[0]
            for p in axs[i].patches:
                percentage = '{:.0f}%'.format(100 * p.get_height()/total)
                x = p.get_x() + p.get_width() + 0.02
                y = p.get_y() + p.get_height()/2
                axs[i].annotate(percentage, (x, y))
    return


#%%#############################################################################
#BOOK########### 3. UNIVARIATE GRAPHING with PRED ##############################
################################################################################

def graph_univar_pred(df_data, list_var, col_predict , var_type):
    """
    Graph univariate for each variable in list compared with numerical prediction (label in dF)
        var_type = 
        "num" if numeric:       Violin Chart
        "cat" if categorical:   Barplot count
    """
    # Setup the graph subplot
    fig, axs = plt.subplots(len(list_var), 1, 
            figsize=set_size(plt_wd, len(list_var), 1))
    fig.subplots_adjust(hspace = 0.3) # Adjust space between rows
    # loop for graphing
    for i, item in enumerate(list_var):
        cond_a = df_data[item].isna() == False
        cond_b = df_data[col_predict] != np.nan
        dF_dummy = df_data.loc[cond_a & cond_b]
        if var_type == "num":
            sns.boxenplot(x=col_predict, y=item, data=dF_dummy, ax = axs[i])
        if var_type == "cat":
            sns.countplot(x=item, hue=col_predict, data=dF_dummy, ax = axs[i])
            # Add the percentage in the graph
            total = dF_dummy.shape[0]
            for p in axs[i].patches:
                percentage = '{:.0f}%'.format(100 * p.get_height()/total)
                x = p.get_x() + p.get_width() + 0.02
                y = p.get_y() + p.get_height()/2
                axs[i].annotate(percentage, (x, y))
    return


def graph_univar_pred2(df_data_, list_var, col_predict, var_type):
    """
    Graph univariate for each variable in list and each value in col_predict (one column per value) 
    NumVar: Histogram
    Repeat histogram for each of col_predict unique value
    """
    
    # --------------- Prepare data and Graph ----------------------------------#
    df_data = df_data_.copy() # Make a copy because CatType transcends the function....
    num_col = df_data[col_predict].nunique() # Nunique sizre
    # Setup the graph subplot
    fig, axs = plt.subplots(len(list_var), num_col + 1, 
            figsize=set_size(plt_wd, len(list_var), num_col + 1), sharex = 'row')
    fig.subplots_adjust(hspace = 0.3) # Adjust space between rows
    
    #---------------- ReType Categories properly ------------------------------#
    # Need to type properly to avoid missing data in sharex
    from pandas.api.types import CategoricalDtype
    if var_type == "cat":
        for col in list_var:
            unique_val = list(df_data[col].unique())
            unique_val = [x for x in unique_val if str(x) != 'nan']
            cat_type = CategoricalDtype(categories=list(unique_val), ordered=True)
            df_data[col] =  df_data[col].astype(cat_type)
    
    # ------------------- Loop for graphing -----------------------------------#
    for j in range(num_col + 1): # Columns
        if j == 0: # graph all data in first column
            cat = "All Data"
            dF_dummy = df_data
        else:  # Select unique values for subsequent columns
            cat = df_data[col_predict].unique()[j-1]
            dF_dummy = df_data.loc[df_data[col_predict] == cat ]
        # Add graph title with count and %
        count = dF_dummy.shape[0]
        pct = count / df_data.shape[0]
        text = "Name: " + str(cat) + "  # Obs:" + str(count) + "  Prop: " + "{0:.0%}".format(pct) 
        axs[0, j].set_title(text, fontsize=16)
        for i, item in enumerate(list_var): # loop on rows
            dF_dummy2 = dF_dummy.loc[dF_dummy[item].isna() == False] # Exclude Nan
            if var_type == "num": # DistPlot for numerical variables
                sns.distplot(dF_dummy2[item], ax = axs[i, j])
            if var_type == "cat":  # Histogram for numerical variables
                sns.countplot(dF_dummy2[item], ax = axs[i, j])
                # Add the percentage in the graph
                total = dF_dummy2.shape[0]
                for p in axs[i, j].patches:
                    percentage = '{:.0f}%'.format(100 * p.get_height()/total)
                    x = p.get_x() + p.get_width() + 0.02
                    y = p.get_y() + p.get_height()/2
                    axs[i, j].annotate(percentage, (x, y))
    return






#%%#############################################################################
#BOOK#################### 4. MULTIVARIATE GRAPHING #############################
################################################################################
#%%---------------------------------- NUM with CAT ----------------------------#
def graph_multvar_numcat(df_data, list_numvar, list_catvar):
    """
    Graph  simple bivariate Num with cat
    NumVar: Violin Chart
    CatVar: Barplot count
    """
    # Setup the graph subplot
    fig, axs = plt.subplots(len(list_numvar), len(list_catvar), 
            figsize=set_size(plt_wd, len(list_numvar), len(list_catvar)))
    fig.subplots_adjust(hspace = 0.3)
    # Graphing loop
    for i_num, item_num in enumerate(list_numvar):
        for i_cat, item_cat in enumerate(list_catvar):
            if item_cat != item_num:
                cond_a = df_data[item_num].isna() == False
                cond_b = df_data[item_cat].isna() == False
                dF_dummy = df_data.loc[cond_a & cond_b]
                sns.boxenplot(x=item_cat, y=item_num, data=dF_dummy, ax=axs[i_num, i_cat])
    return



#%%---------------------------------- Num with Num ----------------------------#
def graph_multvar_numnum(df_data, list_numvar, **kwargs):
    """
    Parameters
    ----------
    df_data: dataframe containing the data to be graphed
    list_numvar: list with the columns name of numerical variables to be graphed
        kwargs: optional hue= Series / name
    Returns
    -------
    None.
    """
    # Setup the graph subplot
    fig, axs = plt.subplots(len(list_numvar), len(list_numvar), 
            figsize=set_size(plt_wd, len(list_numvar), len(list_numvar)))
    fig.subplots_adjust(hspace = 0.3)
    # Graphing loop
    for i_num, item_num in enumerate(list_numvar):
        for i_num2, item_num2 in enumerate(list_numvar):
            if i_num != i_num2:
                cond_a = df_data[item_num].isna() == False
                cond_b = df_data[item_num2].isna() == False
                dF_dummy = df_data.loc[cond_a & cond_b]
                sns.scatterplot(x=item_num, y=item_num2, data=dF_dummy,
                                hue=kwargs.get('hue'), ax=axs[i_num, i_num2 ])
    return


#%%----------------------------------- CAT with CAT ----------------------------#
def graph_multvar_catcat(df_data, list_catvar):
    """
    Graph  simple bivariate Cat with Cat
    Using Balloon Chart
    """
    # Setup the graph subplot
    fig, axs = plt.subplots(len(list_catvar), len(list_catvar), 
            figsize=set_size(plt_wd, len(list_catvar), len(list_catvar)))
    fig.subplots_adjust(hspace = 0.3)
    # Graphing loop
    for i_cat, item_cat in enumerate(list_catvar):
        for i_cat2, item_cat2 in enumerate(list_catvar):
            if i_cat != i_cat2:
                cond_a = df_data[item_cat].isna() == False
                cond_b = df_data[item_cat2].isna() == False
                dF_dummy = df_data.loc[cond_a & cond_b]
                #plt.figure()
                axs[i_cat, i_cat2] = graph_balloon(dF_dummy, item_cat2,
                                item_cat, axs[i_cat, i_cat2])
    return


# ------------------------ Baloon graph for multi cat -------------------------#
def graph_balloon(dF, x, y, axe):
    """
    Prepare balloon chart with dF[x, y]
    Chart is prepared as an Axes object
    
    """
    # Data
    dF_data = dF.groupby([x,y])[y].count()
    dF_data = dF_data.rename("value")
    dF_data = dF_data.reset_index()
    # Calculate the bubble size
    min_size = dF.shape[0]/100
    max_size = dF.shape[0]/2
    # -------- Create the mapping dictionary for the x and y axis -------------#
    if pd.api.types.is_categorical_dtype(dF[x]) == True:
       xax_dict = dict(enumerate(dF[x].cat.categories))
    else:
       xax_dict = dict(enumerate(dF[x].unique()))
    if pd.api.types.is_categorical_dtype(dF[y]) == True:
       yax_dict = dict(enumerate(dF[y].cat.categories))
    else:
       yax_dict = dict(enumerate(dF[y].unique()))
    # -------------------------- Invert dict ----------------------------------#
    xax_dict = {v: k for k, v in xax_dict.items()}
    yax_dict = {v: k for k, v in yax_dict.items()}
    # --------------------------- Map the data --------------------------------#
    dF_data[x + '_'] = dF_data[x].map(xax_dict)
    dF_data[y +'_'] = dF_data[y].map(yax_dict)
    
    # -------------------------------- Graph ----------------------------------#
    sns.scatterplot(x=x + '_', y=y +'_', size='value', data=dF_data,
                sizes=(min_size, max_size), ax = axe, hue='value')
                #, legend="full")
    axe.set_xlim(-1, len(xax_dict))
    axe.set_ylim(-1, len(yax_dict))
    axe.set_xticks(np.arange(len(xax_dict)))
    axe.set_yticks(np.arange(len(yax_dict)))
    axe.set_xticklabels(xax_dict.keys())
    axe.set_yticklabels(yax_dict.keys())
    return axe    



#%%#############################################################################
#BOOK######################### 4.BITS AND BOLTS ################################
################################################################################

"""
# ----------------- Optional save graph
if graph_ouput == True: 
    myplotname = mp_name + '_MP_2AEReturns_Scatt' + picout_extension
    fig.savefig(spath + myplotname)


# ----------------- Graph quantiles
fig, axes = plt.subplots(3, 1, figsize=ipalib.set_size(plt_wd, 3, 1), sharex=True)
l_quantile=[5, 25, 50, 75, 95]

dF_temp = np.percentile(npz_Sim_T['AE_DeathSA'], l_quantile, axis=1)
dF_temp = pd.DataFrame(dF_temp, index=l_quantile)
dF_temp = pd.melt(dF_temp.reset_index(), id_vars='index', var_name='Year', value_name='AE_DeathSA')

sns.lineplot(data=dF_temp, x='Year', y='AE_DeathSA', hue='index', ax=axes[0], color=pal[3])







"""
