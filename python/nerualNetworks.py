# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:57:24 2021

@author: adbb084
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df =  pd.read_csv('C://Users/adbb084/OneDrive - City, University of London/Documents/Github/intro-to-ai-farhan-labi-main/covid_19_indonesia_time_series_all.csv')

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

#Now we have to encode variables since there are Date and String data types
#.astype(str).apply(le.fit_transform)
for column in df_regression.columns:
    df_regression[column] = LabelEncoder().fit(df_regression[column]).transform(df_regression[column])


def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(
        target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)

X,y = to_xy(df_regression, 'New Active Cases')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


model = Sequential()
model.add(Dense(64, input_shape=X[1].shape, activation='relu')) # Hidden 1
model.add(Dense(64,activation='relu')) #Hidden 2
model.add(Dense(1958)) # Output


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,verbose=2,epochs=20)
model.summary()

y_pred = model.predict(X_test)
print("Shape: {}".format(y_pred.shape))
print("Shape: {}".format(y_test.shape))

print("final RMSE =", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))












