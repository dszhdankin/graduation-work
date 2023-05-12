import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
import keras

import numpy as np
import pandas as pd
import os
import time

tf.config.experimental.enable_tensor_float_32_execution(False)

train_data = pd.read_csv('digit-recognizer/train.csv')
test_data = pd.read_csv('digit-recognizer/test.csv')

x_train = train_data.drop(['label'], axis=1)
y_train = train_data['label']
x_test = test_data
scaler = MinMaxScaler()

x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=scaler.feature_names_in_)
x_test = pd.DataFrame(scaler.transform(x_test), columns=scaler.feature_names_in_)

y_train = to_categorical(y_train, num_classes=10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

t0 = time.process_time()
model.fit(x_train, y_train, batch_size=50, epochs=7)
print(time.process_time() - t0)
print(x_train.shape)

# print(tf.reduce_sum(tf.random.normal([1000, 1000])))
# print(tf.config.list_physical_devices('GPU'))
# print(tf.sysconfig.get_build_info())
