# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:27:12 2022

@author: MO_TAREK
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split;
import PIL
import cv2
import os
from sklearn.linear_model import LinearRegression
from keras.callbacks import EarlyStopping




#read the directory
image_dir = Path('Braille Dataset')


#read images with extention .jpg
dir_list = list(image_dir.glob('*.jpg'))


#read images names
name_list = []
for i in dir_list:
    name_list.append(os.path.basename(i)[0])


#open images and puts into a list
images = []
for dir in dir_list:
    I = cv2.imread(str(dir))
    images.append(I)    


#turn both lists in numpy arrays
images_list = np.array(images)
name_list = np.array(name_list).T #transpose - convert columns to rows


#encodes name_list 
le = LabelEncoder()
name_list = le.fit_transform(name_list)


#normalizes image_list
images_list = images_list / 255.0 


#sample
plt.imshow(images_list[1])


#spliting
X_train, X_test, y_train, y_test = train_test_split(images_list, name_list, test_size=0.3, random_state=42)



#Creating model 
model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),   
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.25),   
    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),
    
    keras.layers.Dense(units=576, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(units=288, activation="relu"),

    keras.layers.Dense(units=26, activation="softmax") #output layer
])


model.compile(optimizer="Adam", loss="SparseCategoricalCrossentropy", metrics=["sparse_categorical_accuracy"])


es1 = EarlyStopping(patience=20, monitor="val_sparse_categorical_accuracy", mode="auto")
es2 = EarlyStopping(patience=20, monitor="val_loss", mode="auto")

#The neural network will stop fitting if it gets 20 epochs without converge

history = model.fit(x=X_train, y=y_train, epochs=100, validation_split=0.3, callbacks=[es1, es2])


model.evaluate(X_test, y_test)













































