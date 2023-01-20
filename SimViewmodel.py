#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 09:40:12 2023

@author: kst
"""

from shamir import Shamir
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing

from genDataMult1 import genData
N=10000
p=np.array([1,2,3,4,5])
A=[0,1]
H=[2,3,4]
SS = Shamir(len(p),2,0,10**6,p)
dts = genData(N,p,A,H,SS)


###########################################
## split data
###########################################
dts_train, dts_test = model_selection.train_test_split(dts, 
                      test_size=0.3)

# ## print info
# print("X_train shape:", dts_train.drop("Y",axis=1).shape, "| X_test shape:", dts_test.drop("Y",axis=1).shape)
# print("y_train mean:", round(np.mean(dts_train["Y"]),2), "| y_test mean:", round(np.mean(dts_test["Y"]),2))
# print(dts_train.shape[1], "features:", dts_train.drop("Y",axis=1).columns.to_list())

###########################################
# Normalize data
###########################################
# scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
# X = scaler.fit_transform(dts_train.drop("Y", axis=1))
# dts_train_scaled= pd.DataFrame(X, columns=dts_train.drop("Y", axis=1).columns, index=dts_train.index)
# # dts_scaled["Y"] = dts_train["Y"]

# X = scaler.fit_transform(dts_test.drop("Y", axis=1))
# dts_test_scaled= pd.DataFrame(X, columns=dts_test.drop("Y", axis=1).columns, index=dts_test.index)

###########################################
# Model
###########################################

selected_features = dts.columns #['cA_0', 'cA_1', 'x1x2A_0', 'x1x2A_1']

train, test = dts_train[selected_features], dts_test[selected_features]
label_train, label_test = dts_train['Y'], dts_test['Y']

model_view = tf.keras.models.Sequential([
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2, activation = 'softmax')
])

model_view.compile(loss=sparse_categorical_crossentropy,
          optimizer=Adam(),
          metrics=['accuracy'])

model_view.fit(train, label_train, epochs = 5)

predictions = model_view.predict(test)

predicted_labels = np.argmax(predictions, axis = 1)

compare = (label_test == predicted_labels)

rate = np.sum(compare)/len(predicted_labels)

print(rate)