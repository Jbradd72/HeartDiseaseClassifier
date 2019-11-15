# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:27:30 2019

@author: JeffBradley
"""
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = pd.read_csv('cardio_train.csv', sep=';')

#PreProcessing
y = data['cardio']
x = data.drop('cardio', axis = 1)
colsToFit = ['ap_lo', 'ap_hi', 'height', 'weight']
scaler = MinMaxScaler()

x[colsToFit] = scaler.fit_transform(x[colsToFit])
new_list = []
for i in x['age']:
    years = i / 365
    if (years > 50):
        new_list.append(2)
    elif (years > 35):
        new_list.append(1)
    else:
        new_list.append(0)
    
x['age'] = new_list;
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size = .98, shuffle=True)


classifier = KNeighborsClassifier(n_neighbors=numNeighbors)
classifier.fit(xTrain, yTrain)
predictions = classifier.predict(xTest)

print("CAR: k = " + str(numNeighbors))
print(confusion_matrix(yTest, predictions))
print(classification_report(yTest, predictions))
