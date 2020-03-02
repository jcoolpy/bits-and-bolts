# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 14:54:58 2020
@author: pascal winter
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
sns.set_palette(sns.color_palette("Paired"))
plt_wd = graphlib.plt_wd # set plot width


# ---------------------  for machine learning
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #
from sklearn.linear_model import LogisticRegression #
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

 
#%%#############################################################################
#BOOK###################### INITIALISATION #####################################
################################################################################

csv_mp_outputname = "Output2.csv"

# ----------------------------- Import Dataset --------------------------------#
csv_test = list(CWD.rglob('test.csv'))[0]
dF_Test = pd.read_csv(csv_test)
csv_train = list(CWD.rglob('train.csv'))[0]
dF_Train = pd.read_csv(csv_train)

# --------------------- Type and groom dataset --------------------------------#
# Test
dF_Test = dF_Test.set_index(dF_Test["PassengerId"] )
dF_Test = dF_Test.drop("PassengerId", axis = 1)
dF_Test["Sex"] = dF_Test["Sex"].astype('category')
dF_Test["Embarked"] = dF_Test["Embarked"].astype('category')
# Train
dF_Train = dF_Train.set_index(dF_Train["PassengerId"] )
dF_Train = dF_Train.drop("PassengerId", axis = 1)
dF_Train["Survived"] = dF_Train["Survived"].astype('bool')
dF_Train["Sex"] = dF_Train["Sex"].astype('category')
dF_Train["Embarked"] = dF_Train["Embarked"].astype('category')


#%%#############################################################################
#BOOK################### BASIC EXPLORATION #####################################
################################################################################
# ------------- Calculate a Basic Description  --------------------------------#
dF_des = dalib.data_describe(dF_Train)
dF_des2 = dalib.data_describe(dF_Test)

# -------------- Propose variable classification ------------------------------#
list_numvar, list_catvar = dalib.data_classvars(dF_des)
list_catvar.append('Age')


#%%#############################################################################
#BOOK###########################  GRAPHING #####################################
################################################################################
# Univariate
graphlib.graph_univar(dF_Train, list_numvar, 'num')
graphlib.graph_univar(dF_Train, list_catvar, 'cat')
# Univariate with prediction 
graphlib.graph_univar_pred(dF_Train, list_numvar, 'Survived', 'num')
graphlib.graph_univar_pred(dF_Train, list_catvar, 'Survived', 'cat')
graphlib.graph_univar_pred2(dF_Train, list_numvar, 'Survived', 'num')
graphlib.graph_univar_pred2(dF_Train, list_catvar, 'Survived', 'cat')

# Bivariate Numcat
graphlib.graph_multvar_numcat(dF_Train, list_numvar, list_catvar)
# Bivariate Numnum
graphlib.graph_multvar_numnum(dF_Train, list_numvar, hue = 'Survived')
# Bivariate Catcat
graphlib.graph_multvar_catcat(dF_Train, list_catvar)


#%%#############################################################################
#BOOK########################## CORRELATION ####################################
################################################################################
# ------------------------------ Basic Heatmap --------------------------------#
#fig, ax = plt.subplots(figsize=(12, 8))
#ax.set_ylim(-0.5,len(dF_Train.corr()) + 0.5)
plt.figure(figsize=(12, 8))
sns.heatmap(dF_Train.corr(), annot=True, linewidths=0.2, #ax=ax,
            vmin=-0.5, vmax=0.5, center= 0, cmap='coolwarm')


#%%#############################################################################
#BOOK##################### Feature Engineering #################################
################################################################################
"""
## Pclass: keep as is (1,2,3 is well anti corrlated with ouput)
## SibSp: group: 0 / 1 / 2 / 3-4-5-8 - keep as is
## Sex: 0 / 1
## Parch: group: 0 / 1 / 2 -5-3-4-6 - keep as is
## Embarked: dummy encoding 

# Age: bin
# Fare: log & normalise

######### FILNA
# Embarked: manual - force to S
# Age: categorize by Gender, Sibsp(gpd), Parch(gpd) and Class
"""

# Copy orginal dataframes and create a global one
dF_Train2 = dF_Train.copy()
dF_Test2 = dF_Test.copy()
dF_Test2['Survived'] = np.nan
dF_Train2['Source'] = "Train"
dF_Test2['Source'] = "Test"
dF_All = dF_Train2.append(dF_Test2)

#------------------------ Title -----------------------------------------------#
dF_All["Title"] = dF_All['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) 
# get the dict for mapping
dict_map = dict(enumerate(dF_All['Title'].unique())) # enumerate
dict_map = {v: k for k, v in dict_map.items()} # reversreverse map
dict_map ={'Mr': 'Mr',
             'Mrs': 'Mrs',
             'Miss': 'Miss',
             'Master': 'Master',
             'Don': 'Mr',
             'Rev':  'Spe',
             'Dr': 'Spe',
             'Mme': 'Mrs',
             'Ms': 'Miss',
             'Major': 'Spe',
             'Lady': 'Spe',
             'Sir': 'Mr',
             'Mlle': 'Miss',
             'Col': 'Spe',
             'Capt': 'Spe',
             'Countess': 'Spe',
             'Jonkheer': 'Mr',
             'Dona': 'Mrs'}
# replace
dF_All['Title'] = dF_All['Title'].replace(dict_map)
# Hot encoding
dF_dummy = pd.get_dummies(dF_All['Title'], prefix = 'Title' )
dF_All = pd.concat([dF_All, dF_dummy], axis = 1)


#----------------------- Pclass: no change ------------------------------------#
# Get average   survival: 
dF_dummy = dF_All.loc[dF_All['Survived'].isna() == False]
dF_dummy = dF_All.groupby(['Pclass'])['Survived'].agg(['count', 'mean'])
# seems 1,2,3 is well correlated so keep it like this


#-------------------------------- Sex -----------------------------------------#
# map with 0 and 1
dict_map = dict(enumerate(dF_All['Sex'].unique())) # enumerate
dict_map = {v: k for k, v in dict_map.items()} # reversreverse map
print(dict_map) # print
# replace
dF_All['Sex'] = dF_All['Sex'].replace(dict_map)



#------------------------------ SibSp -----------------------------------------#
# ------------ Group -----------------
# look at the mean for cat over 2
a = dF_All.loc[dF_All['SibSp'] > 2,'SibSp'].mean()
# enumerate
dict_map = dict(enumerate(dF_All['SibSp'].unique())) # enumerate
dict_map = {v: k for k, v in dict_map.items()} # reversreverse map
print(dict_map) # print
# manually adjust mapping
dict_map = {1: 1, 0: 0, 3: 3 , 4: 3, 2: 2, 5: 3, 8: 3}
# replace
dF_All['SibSp'] = dF_All['SibSp'].replace(dict_map)
# look at the mean survical for new encoded variable
dF_dummy = dF_All.loc[dF_All['Survived'].isna() == False]
dF_dummy = dF_dummy.groupby(['SibSp'])['Survived'].agg(['count', 'mean'])


#------------------------------ Parch -----------------------------------------#
# ------------ Group -----------------
# look at the mean for cat over 2
dF_dummy = dF_All.groupby(['Parch'])['Survived'].agg(['count', 'mean'])
# enumerate
dict_map = dict(enumerate(dF_All['Parch'].unique())) # enumerate
dict_map = {v: k for k, v in dict_map.items()} # reversreverse map
print(dict_map) # print
# manually adjust mapping
dict_map = {0: 0, 1: 1, 2: 2, 5: 2, 3: 2, 4: 2, 6: 2, 9:2}
# replace
dF_All['Parch'] = dF_All['Parch'].replace(dict_map)

# look at the mean survical for new encoded variable
dF_dummy = dF_All.loc[dF_All['Survived'].isna() == False]
dF_dummy = dF_All.groupby(['Parch'])['Survived'].agg(['count', 'mean'])


#-------------------------------- Embarked ------------------------------------#
#-------------------------- FillNA -------
# after looking, S is the most likely (Ticket number and fare)
dF_All.loc[dF_All['Embarked'].isna() == True , 'Embarked'] = "S"
#-------------------------- One Hot encoding
dF_dummy = pd.get_dummies(dF_All['Embarked'], prefix = 'Embarked' )
dF_All = pd.concat([dF_All, dF_dummy], axis =1)


#------------------------------------------------------------------------------#
#---------------------------------- Age ---------------------------------------#
#-------------------------- FillNA -------
# Use a grouping  by Gender, SibSp(gpd), Parch(gpd) and Class
dF_dummy = dF_All.groupby(['Parch', 'SibSp', 'Pclass'])['Age'].agg(['count', 'mean'])
dF_All2 = pd.merge(dF_All, dF_dummy, how='left',
                              left_on=['Parch', 'SibSp', 'Pclass'], right_index=True)
# When Nan, assign the result of the mean age
dF_All2['Age_Fill'] = dF_All2['Age']
dF_All2.loc[dF_All2['Age'].isna() == True, 'Age_Fill'] = dF_All2.loc[dF_All2['Age'].isna() == True, 'mean']
# Clean
dF_All2 = dF_All2.drop('count', axis=1)
dF_All2 = dF_All2.drop('mean', axis=1)
dF_All2['Age_G'] = dF_All2['Age_Fill']


#---------------------------- Binning
# bin the category
mybins = [0, 3, 6, 10, 15, 20, 30, 40, 50, 100]
dF_All2['Age_G'] = pd.cut(dF_All2['Age_Fill'], bins=mybins, labels=False)
# calculate the average across the bins and assign it as cat label
dF_dummy = dF_All2.groupby(['Age_G'])['Age'].agg(['count','mean'])
dF_All2 = pd.merge(dF_All2, dF_dummy, how='left',
                              left_on=['Age_G'], right_index=True)

dF_All2['Age_G'] = dF_All2['mean'].astype(int)
# Clean
dF_All2 = dF_All2.drop('count', axis=1)
dF_All2 = dF_All2.drop('mean', axis=1)

dF_All3 = dF_All2.copy()
#----------------- Hot Encode
# dF_dummy = pd.get_dummies(dF_All2['Age_G'], prefix = 'Age' )
# dF_All2 = pd.concat([dF_All2, dF_dummy], axis = 1)

# #----------------- Encode with average success
# dF_dummy = dF_All2.loc[dF_All2['Survived'].isna() == False]
# dF_dummy = dF_All2.groupby(['Age_G'])['Survived'].mean().rename('MeanSurv')
# dF_All3 = pd.merge(dF_All2, dF_dummy, how='left',
#                               left_on=['Age_G'], right_index=True)
# dF_All3['Age_G'] = dF_All3['MeanSurv']
# dF_All3 = dF_All3.drop('MeanSurv', axis =1)


#---------------------------------- Fare --------------------------------------#
# FillNa for Test dataframe
dF_All3.loc[dF_All3['Fare'].isna() == True,'Fare'] = 10
#dF_dummy = dF_Train3.groupby(['Pclass','Sex','SibSp'])['Fare'].agg(['count','mean'])
# Fare: log & normalise
dF_All3['Fare_log'] = np.log(dF_All3['Fare'] + 1)
dF_All3['Fare_log'] = dF_All3['Fare_log'] - dF_All3['Fare_log'].mean()
dF_All3['Fare_log'] = dF_All3['Fare_log'] / dF_All3['Fare_log'].std()
#plt.figure()
#sns.distplot(dF_All3['Fare_log'] )

#------------------------------ Finalise --------------------------------------#
covar =['Pclass', 'Sex', 'Age_G', 'SibSp', 'Parch', 'Fare_log'
        , 'Embarked_C', 'Embarked_Q',
        'Title_Master','Title_Miss', 'Title_Mr', 'Title_Mrs']
var = ['Survived']
dF_TrainFinal = dF_All3.loc[dF_All3['Source'] == "Train", covar + var]
dF_TestFinal = dF_All3.loc[dF_All3['Source'] == "Test", covar]

# Just to have a look...
#dF_des3 = dalib.data_describe(dF_All3)
#dF_des4 = dalib.data_describe(dF_TrainFinal)
#dF_des5 = dalib.data_describe(dF_TestFinal)







#%%#############################################################################
#BOOK##################### MACHINE LEARNING ####################################
################################################################################
# From https://www.kaggle.com/atuljhasg/titanic-top-3-models
# https://www.kaggle.com/rishitdagli/titanic-prediction-using-9-algorithims


# -------------------- Split the data for testing -----------------------------#
X = dF_TrainFinal.drop(['Survived'], axis=1)
y = dF_TrainFinal["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

# -------------------- Initialise the learning --------------------------------#
# Model Names (the size will define the # of models)
lModelName = ['Logistic Regression','Random Forest','GradientBoosting']

# -------------------- Define Model parameters
param = [{} for i in range(len(lModelName))]
# Parameters for logit
param[0] = {'max_iter': 1000,
            'C': 10.0,
            'penalty': 'l2'}
# Parameters for random forest
param[1] = {'max_features': 9,
            'min_samples_leaf': 10,
            'n_estimators': 500}
# Parameter for 
param[2] = {'n_estimators': 10,
            'learning_rate': 0.1,
            'max_depth': 6,
            'max_features': 1.0,
            'min_samples_leaf': 20}

# ------------------ Define Models
models = [LogisticRegression(**param[0]),
          RandomForestClassifier(**param[1]),
          GradientBoostingClassifier(**param[2])]

# ------------------ Define Score Train results
dF_ModelScore = pd.DataFrame({
        'Model': np.arange(len(models)),
        'Train Score': np.zeros(len(models)),
        'CVal Score': np.zeros(len(models))})
# Define Model features results
dF_ModelFeat = pd.DataFrame(index = X_train.columns)


# ------------------------------- TRAIN THE MODELS ----------------------------#
for i in range(len(models)):
    # Train the model
    models[i].fit(X_train,y_train)
    # Log the training score and cross val score
    dF_ModelScore.loc[i,:] = [lModelName[i], 
                      models[i].score(X_train,y_train),  
                      accuracy_score(y_val, models[i].predict(X_val))]
    #print(classification_report(y_val, models[1].predict(X_val)))



# ----------------------------- ASSESS THE MODELS -----------------------------#
# ----------- Assess the models by "Bootstrapping" them on various subsets
dF_ModelAssess , dF_ModelAccuracy = dalib.model_assessment(lModelName, models, X, y, 5)

# ----------- Get feature importance (cannot loop since logit is different)
dF_ModelFeat[lModelName[0]] = abs(models[0].coef_[0])
dF_ModelFeat[lModelName[0]]  = dF_ModelFeat[lModelName[0]] /dF_ModelFeat[lModelName[0]].sum()
dF_ModelFeat[lModelName[1]] = models[1].feature_importances_
dF_ModelFeat[lModelName[2]] = models[2].feature_importances_
# Graph
dF_ModelFeat2 = dF_ModelFeat.stack().reset_index().rename(columns = {"level_0": "Feat", "level_1": "Model", 0: "Value"})
plt.figure() 
sns.catplot(data=dF_ModelFeat2, x="Feat", y = "Value", hue="Model", kind ='bar').set_xticklabels(rotation=45)







#%%#############################################################################
#BOOK##################### FINAL OUTPUT ########################################
################################################################################

#---------------------------------- Output ------------------------------------#
# Retrain the model with whole Train population
#model2.fit(X, y)
dF_TestFinal2 = dF_TestFinal.copy()
dF_TestFinal2['Survived'] = models[1].predict(dF_TestFinal)
dF_Output = dF_TestFinal2['Survived'].astype('int32')

#------------ Only female surviving
#dF_Output = (dF_Test['Sex'] == 'female').astype('int32')
#dF_Output.name = 'Survived'


# Export
spath = csv_test.parent.as_posix() + "/" + csv_mp_outputname
dF_Output.to_csv(spath, header = True) #, date_format='%Y%m%d')


