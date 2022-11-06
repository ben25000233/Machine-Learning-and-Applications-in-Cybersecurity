#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda
from siamese import SiameseNetwork
import numpy as np
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# In[3]:


batch_size = 3
epochs = 20

# input image dimensions
img_rows, img_cols = 10, 16


# In[4]:


df = pd.read_csv(
    '../dataset.csv', low_memory=False)
df = df[['filename', 'label']]
labelcsv = df[(df['label'] == 'Mirai') | (
    df['label'] == 'Bashlite') | (df['label'] == 'Unknown')]

# In[1]:


def read_data(path):
    train = []
    label = []
    files = os.listdir(path)
    for file in files:
        filename = file.replace('.jpg', '')
        match = labelcsv[labelcsv['filename'] == filename]
        if(not match.empty):
            label.append(match.iat[0, 1])
            img = np.array(Image.open(path+'/' +
                                      file))
            train.append(img)
    train = np.array(train)
    print(train.shape)
    return train, label


# In[ ]:


path = '../img_data/train'
x_train, y_train = read_data(path)

path = '../img_data/test'
x_test, y_test = read_data(path)

labelencoder = LabelEncoder()
labelencoder.fit(y_train)
y_train = labelencoder.transform(y_train)
y_test = labelencoder.transform(y_test)

input_shape = (img_rows, img_cols, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[27]:


def create_base_model(input_shape):
    model_input = Input(shape=input_shape)

    embedding = Conv2D(32, kernel_size=(
        3, 3), input_shape=input_shape)(model_input)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Conv2D(64, kernel_size=(3, 3))(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Flatten()(embedding)
    embedding = Dense(128)(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)

    return Model(model_input, embedding)


def create_head_model(embedding_shape):
    embedding_a = Input(shape=embedding_shape[1:])
    embedding_b = Input(shape=embedding_shape[1:])
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([embedding_a, embedding_b])
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    return Model([embedding_a, embedding_b], prediction)


# In[9]:


base_model = create_base_model(input_shape)
head_model = create_head_model(base_model.output_shape)

siamese_network = SiameseNetwork(base_model, head_model)
siamese_network.compile(loss='binary_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

siamese_checkpoint_path = "./siamese_checkpoint"

siamese_callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=0),
    ModelCheckpoint(siamese_checkpoint_path,
                    monitor='val_accuracy', save_best_only=True, verbose=0)
]


# In[10]:


# In[11]:
scores = []
siamese_network.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=siamese_callbacks)
score = siamese_network.evaluate(x_test, y_test, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])