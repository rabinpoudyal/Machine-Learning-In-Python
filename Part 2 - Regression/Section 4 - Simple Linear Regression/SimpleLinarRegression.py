#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:21:09 2018

@author: rabin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')
print(data.head())


X = data.iloc[:, :-1].values # all rows except last
Y = data.iloc[:, 1].values # all rows of 3rd column

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
'''
# Sometimes we need feature scaling too when the algorithm does not take care of scaling
# but in our case it does so we dont need to use it
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
'''

# Now fit the simple linear regression to our training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the salary of employees of test set and compare it with real salary
Y_predicted = regressor.predict(X_test)

# Visualize the prediction in training set

plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue") # This line is givin by the regressor
plt.title("Salary vs experience")
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.show()

# Visualize the prediction in test set

plt.scatter(X_test, Y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue") # We dont need to change this because the line is what we predicted earlier
plt.title("Salary vs experience")
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.show()