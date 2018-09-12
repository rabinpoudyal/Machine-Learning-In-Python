#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 07:33:44 2018

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

# Lets build linear regression and polynomial regression model and see which works best

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)

# Ploynomial regressor

from sklearn.preprocessing import PolynomialFeatures
# Transform our feature vector X into polynomial features
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)
poly_regressor.fit(X_poly, Y)
# Use those polynomial features to fit linear regression
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly, Y)

# Visualize the linear regression 
plt.scatter(X, Y)
plt.plot(X, linear_regressor.predict(X))
plt.title("Using Linear Regression model")
plt.xlabel("Employee Label")
plt.ylabel("Expected Salary")
plt.show()

# Visualize the polynomial regression 
plt.scatter(X, Y)
plt.plot(X, linear_regressor2.predict(poly_regressor.fit_transform(X)))
plt.title("Using Ploynomial Regression model")
plt.xlabel("Employee Label")
plt.ylabel("Expected Salary")
plt.show()

# Now let's predict the salary of employee of level 6.5
# Using linear regression
linear_regressor.predict(6.5)
# Using polynomial regression
linear_regressor2.predict(poly_regressor.fit_transform(6.5))