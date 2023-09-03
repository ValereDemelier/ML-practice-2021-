#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:33:23 2021

@author: valeredemelier
"""

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import matplotlib.pyplot as plt
seed = 123
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
mse = MSE(y_test, y_pred)
rmse = mse**(1/2)
r2 = r2_score(y_test, y_pred)
print('A Simple linear regression model has a RMSE of: {} \nand a R2 score of: {}\n'
      .format(rmse, r2))

neural = MLPRegressor(random_state=seed, max_iter=1000)
neural.fit(X_train, y_train)
neural_pred = neural.predict(X_test)
rmse_neur = MSE(y_test, neural_pred)**(1/2)
neural_r2 = r2_score(y_test, neural_pred)
print('A Nueral Network Regression model has a RMSE of: {} \nand a R2 score of: {}\n'
      .format(rmse_neur, neural_r2))

dt = DecisionTreeRegressor(max_depth=10, random_state=seed)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_rsme = MSE(y_test, dt_pred)**(1/2)
dt_r2 = r2_score(y_test, dt_pred)
print('A Decision Tree Regressor model has a RMSE of: {} \nand a R2 score of: {}\n'
      .format(dt_rsme, dt_r2))

rf = RandomForestRegressor(n_estimators=500, random_state=seed)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = MSE(y_test, rf_pred)**(1/2)
rf_r2 = r2_score(y_test,rf_pred)
print('A Random Forest Regressor model has a RMSE of: {} \nand a R2 score of: {}\n'
      .format(rf_rmse, rf_r2))

vr = VotingRegressor(estimators = [('lin', lin_reg), ('dt',dt), ('Neural',neural), ('rf',rf)])
vr.fit(X_train, y_train)
vr_pred = vr.predict(X_test)
vr_rmse = MSE(y_test, vr_pred)**(1/2)
vr_r2 = r2_score(y_test, vr_pred)
print('An Ensemble Voting Regressor model has a RMSE of: {} \nand a R2 score of: {}\n'
      .format(vr_rmse, vr_r2))


_ = plt.plot(X_test, y_test, linestyle='none', marker='o', color='b', alpha=.1)
_ = plt.plot(X_test, y_pred, linestyle='none', marker='x', color='r', alpha =.1)
_ = plt.plot(X_test, vr_pred, linestyle='none', marker='v', color='y', alpha=.1)
plt.ylabel('Boston Housing Prices')
plt.title('Linear Regression vs Voting Regressor')









