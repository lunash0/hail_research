import os
import sys
import glob
import random
import warnings
import itertools
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
# import seaborn as sns
from collections import Counter
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree


# import keras
# https://inpages.tistory.com/156
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Conv2D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.python.client import device_lib

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
# warnings.filterwarnings(action='ignore')
# warnings.filterwarnings(action='default')
#

# print(device_lib.list_local_devices())
# keras.backend.tensorflow_backend._get_available_gpus()


CIFAR100_CLASSES = sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',  # aquatic mammals
                           'aquarium' 'fish', 'flatfish', 'ray', 'shark', 'trout',  # fish
                           'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',  # flowers
                           'bottles', 'bowls', 'cans', 'cups', 'plates',  # food containers
                           'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',  # fruit and vegetables
                           'clock', 'computer' 'keyboard', 'lamp', 'telephone', 'television',  # household electrical devices
                           'bed', 'chair', 'couch', 'table', 'wardrobe',  # household furniture
                           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',  # insects
                           'bear', 'leopard', 'lion', 'tiger', 'wolf',  # large carnivores
                           'bridge', 'castle', 'house', 'road', 'skyscraper',  # large man-made outdoor things
                           'cloud', 'forest', 'mountain', 'plain', 'sea',  # large natural outdoor scenes
                           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',  # large omnivores and herbivores
                           'fox', 'porcupine', 'possum', 'raccoon', 'skunk',  # medium-sized mammals
                           'crab', 'lobster', 'snail', 'spider', 'worm', # non-insect invertebrates
                           'baby', 'boy', 'girl', 'man', 'woman',  # people
                           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',  # reptiles
                           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',  # small mammals
                           'maple', 'oak', 'palm', 'pine', 'willow',  # trees
                           'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',  # vehicles 1
                           'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'  # vehicles 2
                          ])



(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=0)

# print('size of CIFAR-100 Train data : {}'.format(len(X_train)))  # 40000
# print('size of CIFAR-100 Validation data : {}'.format(len(X_valid)))  # 10000
# print('size of CIFAR-100 Test data : {}'.format(len(X_test)))  # 10000

print("X_train Shape : ", X_train.shape)  # (40000, 32, 32, 3)
print("X_train Type : ", type(X_train))  # <class 'numpy.ndarray'>
print("y_train Shape : ", y_train.shape)  # (40000, 1)
print("y_train Type : ", type(X_train))  # <class 'numpy.ndarray'>

print("X_val Shape : ", X_valid.shape)  # (10000, 32, 32, 3)
print("y_val Shape : ", y_valid.shape)  # (10000, 1)

x_train_0 = X_train[0]
print(x_train_0.shape)  # (32, 32, 3)
# print(x_train_0.dtype)  #
# print(y_train.dtype)
print(y_train)

# Reshape Data
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_valid = X_valid.reshape(X_valid.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

print("\n")
print(X_train.shape)

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255
X_valid = X_valid / 255
X_test = X_test / 255

y_train = to_categorical(y_train, 100)
y_valid = to_categorical(y_valid, 100)
y_test = to_categorical(y_test, 100)


INPUT_SHAPE = (32, 32, 3)

# base_model = ResNet50(input_shape=INPUT_SHAPE,
#                       include_top=False,
#                       weights='imagenet')
#
# base_model.summary()
# base_model.trainable = True
# set_trainable = False
#
# for layer in tqdm(base_model.layers):
#     if layer.name in ['res5a_branch2a', 'res5a_branch2b', 'res5a_branch2c',
#                       'res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c',
#                       'res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c']:
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
#
# layers = [(layer, layer.name, layer.trainable) for layer in base_model.layers]
# # layers_df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
# # print(layers_df)
#
#
# model = Sequential()
#
# model.add(base_model)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(len(CIFAR100_CLASSES), activation='softmax'))
#
# model.compile(loss=categorical_crossentropy,
#               optimizer=Adam(learning_rate=0.0001),
#               metrics=['acc', 'top_k_categorical_accuracy'])
#
# model.summary()
#
# EPOCHS = 10
# BATCH_SIZE = 64
#
# history = model.fit(X_train,
#                     y_train,
#                     epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     use_multiprocessing=True,
#                     validation_data=(X_valid, y_valid)
#                    )
#
# model.compile(loss=categorical_crossentropy,
#               optimizer=Adam(learning_rate=0.0001),
#               metrics=['acc','top_k_categorical_accuracy'])