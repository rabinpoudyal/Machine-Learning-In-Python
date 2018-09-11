#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 15:44:00 2018

@author: rabin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Data.csv', delimiter=",")
print(data.head())

X = data.iloc[:, :-1].values # all rows except last

Y = data.iloc[:, 3].values # all rows of 3rd column

# Taking care of nan values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

# Fit to the columns that has missing values
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Now Lets encode the categorical variable country and purchased

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# We don't need to perform one hot encoding because it is a dependednt variable and
# ML model will know there is no relative order between two
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Spliting the dataset into test and train set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature scaling to remove dominant behaviour, since one variable dominates another
# variable 

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# We dont need to feature scale independent variable if it is classification but should apply
# if it is regresssion in large dependent variable

