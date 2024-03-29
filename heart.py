# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:27:30 2019

@author: JeffBradley
"""
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

def doRandomForest(iter):
    print(iter)
    rfc = RandomForestClassifier(n_estimators=iter, max_depth=2,
                                  random_state=0)
    rfc.fit(xTrain, yTrain)
    predictions = rfc.predict(xTest)
    score = accuracy_score(yTest, predictions)
    return iter, score

data = pd.read_csv('cardio_train.csv', sep=';')

#PreProcessing
y = data['cardio']
x = data.drop('cardio', axis = 1)
x = x.drop('id', axis = 1)
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
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size = .9, shuffle=True)


best_score = 0;
best_i = 0
best_j = 1
best_rfc = RandomForestClassifier(n_estimators=1, max_depth=1,
                                      random_state=0)
scores = []
n = range(1,100)
for i in tqdm(n):
    for j in range(2,3):
        rfc = RandomForestClassifier(n_estimators=i, max_depth=j,
                                      random_state=0)
        rfc.fit(x, y)
        predictions = rfc.predict(xTest)
        score = accuracy_score(yTest, predictions)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_i = i
            best_j = j
            best_rfc = rfc


print(best_i, best_j, best_score)
plt.plot(n, scores)

joblib.dump(best_rfc, "./best_forest")


best_rfc = joblib.load("./best_forest")
#final_pred = best_rfc.predict(xTest)
#ngb = GaussianNB()
#ngb.fit(xTrain, yTrain)

print(best_rfc.n_estimators, best_rfc.max_depth)

'''for i in range(len(final_pred)):
    pred = final_pred[i]
    if pred == 1:
        prob = ngb.predict_proba(xTest.to_numpy()[i].reshape(1,-1))[0][1] * 100
        if prob > 50:
            print("You were slain by Thanos, for the good of the Universe.", prob)
        #else:
            #print("You may be at risk for heart disease, you should consult with a healthcare professional")
        
    else:
        
        prob = ngb.predict_proba(xTest.to_numpy()[i].reshape(1,-1))[0][0] * 100
        if prob > 50:
            print("Thanos has spared you", prob)'''
        #else:
            #print("You may be at risk for heart disease, you should consult with a healthcare professional")
#print(confusion_matrix(yTest, predictions))
#print(classification_report(yTest, predictions))

#print("RFC accuracy: ", accuracy_score(yTest, predictions))
