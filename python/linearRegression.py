# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:16:52 2021

@author: adbb084
"""

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



#This is just whats ready locally on my machine
df =  pd.read_csv('C://Users/adbb084/OneDrive - City, University of London/Documents/Github/intro-to-ai-farhan-labi-main/covid_19_indonesia_time_series_all.csv')
                  
                  
#See what dataframe looks like
df


#Checking for any null values
print(df.isnull().any())


#Create a copy of the dataset to work on
df_regression = df.copy()


#For the NA values that are in int columns, fill them with medians 
med_gf_nc = df_regression['Growth Factor of New Cases'].median()
med_gf_nd = df_regression['Growth Factor of New Deaths'].median()
med_tot_uv = df_regression['Total Urban Villages'].median()
med_tot_rv = df_regression['Total Rural Villages'].median()
med_tot_c = df_regression['Total Cities'].median()

df_regression['Growth Factor of New Cases'] = df_regression['Growth Factor of New Cases'].fillna(med_gf_nc)

df_regression['Growth Factor of New Deaths'] = df_regression['Growth Factor of New Deaths'].fillna(med_gf_nd)

df_regression['Total Urban Villages'] = df_regression['Total Urban Villages'].fillna(med_tot_uv)

df_regression['Total Rural Villages'] = df_regression['Total Rural Villages'].fillna(med_tot_rv)

df_regression['Total Cities'] = df_regression['Total Cities'].fillna(med_tot_c)


#We can drop the columns: City or Regency, Province, Island, Time Zone, Special Status.
#This is because we already have Location and Location ISO code, for the whole of Indonesia
df_regression = df_regression.drop(columns=['City or Regency', 'Province', 'Island', 'Time Zone', 'Special Status'])


#Check what df looks like now
df_regression 


#Check for any null values now
print(df_regression.isnull().any())


#Now we have to encode variables since there are Date and String data types
#.astype(str).apply(le.fit_transform)
for column in df_regression.columns:
    df_regression[column] = LabelEncoder().fit(df_regression[column]).transform(df_regression[column])



#Check types of df
df_regression.dtypes
df_regression


#Spltting test train, where we are predicting the number of new cases, where y is target and x is everything else
X = df_regression.drop(columns=['New Active Cases'])
y = df_regression['New Active Cases']

#Using standard test size of 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#Building linear regression model and fitting
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


#Calculate predictions of model
y_pred = lr_model.predict(X_test)


#Compare predicted vs actual we are looking only at 25 cases here
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)


#Plot model of predicted vs actual with seaborn
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.regplot(x='Actual', y='Predicted', data=df_compare);


#Evaluate model performance using RMSE, R2
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 score: ', metrics.r2_score(y_test, y_pred))


#try with cross validation
kf = KFold (5)

fold = 1 

n = 0
for train_index, test_index in kf.split(X,y):
    print(str(n+1) + " This is loop number: " + str(train_index))
    lr_model.fit(X[train_index], y[train_index])
    print("This is the " + str(n))
    y_test1 = y[test_index]
    y_pred1 = lr_model.predict(X[test_index])
    print("Fold", {fold})
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test1, y_pred1)))










