# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:55:35 2019

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


data = pd.read_csv('cardio_train.csv', sep=';')
dx = data.drop('cardio', axis = 1)
dx = dx.drop('id', axis = 1)
y = data['cardio']

ngb = GaussianNB()
ngb.fit(dx, y)

colsToFit = ['ap_lo', 'ap_hi', 'height', 'weight']
scaler = MinMaxScaler()

best_rfc = joblib.load("./best_forest")


x = {};
a =  np.array(int(input("Enter your age (years): ")))
x['age'] = a
x['gender'] = int(input("Enter your gender (1: F, 2: M): "))
scaler.fit(dx['height'].to_numpy().reshape(-1, 1))
h = int(input("Enter your height (cm): "))
x['height'] = scaler.transform(np.array(h).reshape(1,-1))[0]
scaler.fit(dx['weight'].to_numpy().reshape(-1, 1))
w = int(input("Enter your weight (kg): "))
x['weight'] = scaler.transform(np.array(w).reshape(1,-1))[0]
scaler.fit(dx['ap_hi'].to_numpy().reshape(-1, 1))
ah = int(input("Enter your ap_hi: "))
x['ap_hi'] = scaler.transform(np.array(ah).reshape(1, -1))[0]
scaler.fit(dx['ap_lo'].to_numpy().reshape(-1, 1))
al = int(input("Enter your ap_lo: "))
x['ap_lo'] = scaler.transform(np.array(al).reshape(1, -1))[0]
x['cholesterol'] = int(input("Enter your cholesterol (1: normal, 2: above normal, 3: well above normal): "))
x['gluc'] = int(input("Enter your glucose level ( 1: normal, 2: above normal, 3: well above normal): ") )
x['active'] = int(input("Are you active? (1: yes, 2: no): "))
x['smoke'] = 0
x['alco'] = 0

#final_pred = best_rfc.predict(xTest)
#ngb = GaussianNB()
#ngb.fit(xTrain, yTrain)

df = pd.DataFrame(x);


pred = best_rfc.predict(df)
print(pred)

if pred == 1:
    
    prob = ngb.predict_proba(df)[0][1] * 100
    if prob > 150:
        print("Naive Bayes and The Forest agree that you do have heart disease with respective probabilities of ", prob, " and ", best_rfc.predict_proba(df)[0][1] * 100)
    else:
        print("The forest predicts you do have heart disease, but Naive Bayes disagrees.\nYou may be at risk for heart disease, you should consult with a healthcare professional.")
        print("Naive bays predicts you do not have heart disease with a probability of ", 100 - prob)
        print("The random forest predicts you do have heart disease with a probability of ",best_rfc.predict_proba(df)[0][1] * 100)
    
else: 
    prob = ngb.predict_proba(df)[0][0] * 100
    if prob > 50:
        print("Naive Bayes and The Forest agree that you do not have heart disease with respective probabilities of ", prob, " and ", best_rfc.predict_proba(df)[0][0] * 100)
    else:
        print("The forest predicts you do not have heart disease, but Naive Bayes disagrees.\nYou may be at risk for heart disease, you should consult with a healthcare professional.")
        print("Naive bays predicts you do have heart disease with a probability of ", prob)
        print("The random forest predicts you do not have heart disease with a probability of ",100 - best_rfc.predict_proba(df)[0][1] * 100)