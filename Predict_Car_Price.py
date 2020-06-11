# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:52:12 2020

@author: ssark
"""


import pandas as pd
import numpy as np
import pickle


colnames = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 
        'city-mpg', 'highway-mpg', 'price']


cars = pd.read_csv("imports-85.data", names=colnames)

to_drop = ["symboling", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", 
                  "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system", "engine-size"]

cars_num = cars.drop(to_drop, axis=1)

cars_num = cars_num.replace("?", np.nan)

cars_num = cars_num.astype("float")

cars_num = cars_num.dropna(subset=["price"])
cars_num = cars_num.fillna(cars_num.mean())

to_drop_more = ["normalized-losses","wheel-base"]

cars_num = cars_num.drop(to_drop_more, axis=1)

X = cars_num.iloc[:, :-1]

y = cars_num.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Saving model to disk
pickle.dump(regressor, open('predictCarPrice.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('predictCarPrice.pkl','rb'))














