# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split 
#import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#This is just whats ready locally on my machine
df =  pd.read_csv('C://Users/adbb084/OneDrive - City, University of London/Documents/Github/intro-to-ai-farhan-labi-main/covid_19_indonesia_time_series_all.csv')

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

#Encode variables
for column in df_regression.columns:
    df_regression[column] = LabelEncoder().fit(df_regression[column]).transform(df_regression[column])

#X y values
X = df_regression.drop(columns=['New Active Cases']) 
y = df_regression['New Active Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)


rmse_data = []
r2_data = []
nums = []
for i in range(1,75):
    rf_model = RandomForestClassifier(n_estimators=i,criterion="entropy")
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    rmse_data.append(rmse)
    r2_data.append(r2)
    nums.append(i)
    print("This loop number is :" + str(i))


#Plotting n estimator vs r2 & n estimator vs rmse
plt.plot(nums,rmse_data)
plt.xlabel("Number of Trees")
plt.ylabel("RMSE")
plt.show()

plt.plot(nums,r2_data)
plt.xlabel("Number of Trees")
plt.ylabel("R2")
plt.show()





