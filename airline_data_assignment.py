#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:15:58 2021

@author: valeredemelier
Ass_2
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

airline_df = pd.read_csv('/Users/valeredemelier/Documents/Assignment_2.csv',)

def describe_data(data):
    print(data.info())
    for i in data.columns:
        print(data[i].describe())
        print(data[i].value_counts())
        print(data[i].head())

#Replace empty strings with missing values
airline_df = airline_df.replace(r'^\s*$', np.nan, regex=True)

#Convert all dtype object columns to dtype float
for i in airline_df.columns:
    airline_df[i] = airline_df[i].astype(str).astype(float)
        
# Identify invalid values and remove convert them to np.nan

likert_cols = ['Type_of_Travel', 'Class', 'Inflight_wifi_service',
       'Arrival_time_convenient', 'Ease_of_Online_booking', 'Gate_location',
       'Food_and_drink', 'boarding_process', 'Seat_comfort', 'Onboard_service',
       'Leg_room', 'Baggage_handling', 'Checkin_service', 'Cleanliness']
flight_exp_df = airline_df[likert_cols]

flight_exp_df = flight_exp_df.replace([6,7,8,9,10], np.nan, regex=True)

# Impute missing data using sklearn Iterative Imputer :

imp = IterativeImputer(max_iter=10, random_state=42)
imp.fit(airline_df)
imputed_df = np.round(imp.transform(airline_df))
imputed_df = pd.DataFrame(imputed_df)
imputed_df.columns = airline_df.columns

#Scale DataFrame (Except Next Flight Distance?):

X = imputed_df.drop('Next_Flight_Distance', axis =1)
X_1 = StandardScaler().fit_transform(X)
X_1 = pd.DataFrame(X)
X_1.columns = X.columns

# PCA

pca = PCA(n_components=3)
factors = pca.fit_transform(X_1)


# Split DataFrame & predict (try using cross val scoring)"

y = imputed_df['Next_Flight_Distance']
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=.3)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
'''
y = imputed_df['Next_Flight_Distance']
X_train, X_test, y_train, y_test = train_test_split(factors, y, test_size=.3)
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
'''
#Evaluate Model
print(linreg.coef_)

print('The R2 Score is: {}'.format(r2_score(y_test, y_pred)))
print('The mean squared error is: {}'.format(mean_squared_error(y_test, y_pred)))
print('The Explained Variance Score is: {}'.format(explained_variance_score(y_test, y_pred)))


