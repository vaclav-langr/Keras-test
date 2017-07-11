from __future__ import print_function

__author__ = 'vaclavlangr'

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scipy import misc

layers = 4
data_path = ""


def load_data(path):
    from os import listdir
    from random import shuffle

    files = []
    for number in range(10):
        for f in listdir(path + str(number)):
            files.append([number, f])
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

train_data = train_data.reshape((60000, 784))
test_data = test_data.reshape((10000, 784))

train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

model = Sequential()
for i in range(layers):
    model.add(Dense(784, activation='sigmoid', input_shape=(784,)))
    model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))