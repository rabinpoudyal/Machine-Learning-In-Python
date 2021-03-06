#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:05:33 2018

@author: rabin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
print(data.head())

X = data.iloc[:, 1:2].values # just to treat features as matrix since we just have one column
Y = data.iloc[:, 2].values # all rows of 2nd col

"""
# Not required to do any steps below
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
"""

# Ploynomial regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

# Now let's predict the salary of employee of level 6.5
# Using non-linear regression
Y_predicted = regressor.predict(6.5)

# Visualize the polynomial regression 
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color="blue")
plt.plot(X_grid, regressor.predict(X_grid), color="red")
plt.title("Using Random Forest Regression model")
plt.xlabel("Employee Label")
plt.ylabel("Expected Salary")
plt.show()