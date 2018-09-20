#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:41:24 2018

@author: rabin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Restaurant_Reviews.tsv', delimiter="\t", quoting = 3)

# Clean the text and remove things that does not matter
import re
# Remove the stop words
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Create stemmer object - removing same version of words
ps = PorterStemmer()
corpous = []
for i in data.index:
    # First split the sentence so you can loop through each of them
    review = re.sub("[^a-zA-Z$]", " ", data['Review'][i])
    # Transofrm all letters to lowercase
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpous.append(review)

# Create a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpous).toarray()
y = data.iloc[:, 1].values

# Create test set and train set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train the model using naive bayes classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
# Test the accuracy using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print((55+91)/200)

# Trying the clustering algorithm
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, init="k-means++")
y_kmeans = kmeans.fit_predict(X_train)
