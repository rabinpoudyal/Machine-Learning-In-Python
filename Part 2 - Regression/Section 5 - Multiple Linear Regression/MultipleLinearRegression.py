#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:40:05 2018

@author: rabin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
print(data.head())

X = data.iloc[:, :-1].values # all rows except last
Y = data.iloc[:, 4].values # all rows of 3rd column

# Encode the state variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap -> Remove one dummy variable from the model
X = X[:, 1:] # Take all rows starting from column 1 don't take 0th col

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

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

# Predict for the test set

Y_predicted = regressor.predict(X_test)
