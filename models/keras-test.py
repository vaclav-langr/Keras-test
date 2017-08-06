from __future__ import print_function

__author__ = 'vaclavlangr'

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from scipy import misc

layers = 1
data_path = ""
n = 3
m = 2
k = 2


def load_data(path):
    from os import listdir
    from random import shuffle

    files = []
    for number in range(10):
        for file in listdir(path + str(number)):
            files.append([number, file])
    shuffle(files)
    labels, file_name = zip(*files)
    result = np.empty([len(file_name), 28, 28])
    for number in range(len(file_name)):
        img = misc.imread(path + str(labels[number]) + "\\" + file_name[number])
        result[number, :, :] = img
    result = result.astype('float32')
    result /= 255
    return result, np.asarray(labels)

train_data, train_labels = load_data(data_path + "train\\")
test_data, test_labels = load_data(data_path + "test\\")

train_data.resize((train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
test_data.resize((test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

model = Sequential()
for i in range(0, m):
    for j in range(0, n):
        if i == 0 and j == 0:
            model.add(Conv2D(28, (3, 3), padding='same', input_shape=train_data.shape[1:],
                             use_bias=True, bias_initializer='zeros'))
        else:
            model.add(Conv2D(28, (3, 3), use_bias=True, bias_initializer='zeros'))
        model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

model.add(Flatten())

for i in range(0, k):
    model.add(Dense(1024, activation='relu', use_bias=True, bias_initializer='zeros'))
    model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

model.fit(train_data, train_labels, epochs=10, batch_size=128, validation_data=(test_data, test_labels))

model.save_weights('model.hdf5')
with open('model.json', 'w') as f:
    f.write(model.to_json())