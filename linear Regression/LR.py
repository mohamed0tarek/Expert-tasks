# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:07:25 2022

@author: MO_TAREK
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Advertising.csv")
data.head()

data.drop(['Unnamed: 0'], axis=1)

X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X, y)

predictions = reg.predict(X)

plt.figure(figsize=(16, 8))
plt.scatter(
    data['TV'],
    data['sales'],
    c='black')

plt.plot(
    data['TV'],
    predictions,
    c='red',
    linewidth=2)

plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()