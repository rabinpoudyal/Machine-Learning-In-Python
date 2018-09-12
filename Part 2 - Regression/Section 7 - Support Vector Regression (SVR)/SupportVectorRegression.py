#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 08:55:42 2018

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
"""
# SVR model is less common and it does not perform feature scaling so perform it
from sklearn.preprocessing import StandardScaler
# We are going to fit and transform so we need 2 objects if we dont create 2 
# then it will get dirty when used by Y
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

# Support Vector Regressor( SVR ) Regression
from sklearn.svm import SVR
# rbf kernel makes non linear model. there are also linear models here in svr
regressor = SVR(kernel='rbf')
regressor.fit(X,Y) 


# Now let's predict the salary of employee of level 6.5
# Using non-linear regression
Y_predicted = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualize the polynomial regression 
plt.scatter(X, Y, color="blue")
plt.plot(X, regressor.predict(X), color="red")
plt.title("Using SVR Regression model")
plt.xlabel("Employee Label")
plt.ylabel("Expected Salary")
plt.show()
