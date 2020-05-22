#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:55:47 2020

@author: arko
"""

#import libraries
import cv2
import numpy as np
import os
from keras.layers import Activation, Conv2D, GlobalAveragePooling2D, Dropout
from keras.models import Sequential
from keras_squeezenet import SqueezeNet
from keras.utils import to_categorical
from keras.optimizers import Adam

#dictionary of class names
IMG_SAVE_PATH = 'data'

CLASS_NAMES = {
    'rock' : 0,
    'paper' : 1,
    'scissor' : 2,
    'lizard' : 3,
    'spock' : 4,
    'none' : 5
    }

#map each class name
NUM_CLASSES = len(CLASS_NAMES)
def mapper(map):
    return CLASS_NAMES[map]

#create model
def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(227,227, 3), include_top=False),
        Dropout(0.5),
        Conv2D(NUM_CLASSES, (1,1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
        ])
    return model


#load images into one list
#first IMG_SAVE_PATH then CLASS_PATH 

dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    
    for item in os.listdir(path):
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227,227))
        dataset.append([img, directory])
    
dataset[:30]

#one-hot encodation
data, labels = zip(*dataset)
labels = list(map(mapper, labels))
labels = to_categorical(labels)
#compile model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#train model
model.fit(np.array(data), np.array(labels), epochs=10)

#save model
model.save('rock-paper-game.h5')
