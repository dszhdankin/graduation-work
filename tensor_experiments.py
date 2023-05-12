import numpy as np
import tensorflow as tf

import keras

tf.config.experimental.enable_tensor_float_32_execution(False)


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        print(self.w)
        print(self.b)
        return tf.matmul(inputs, self.w) + self.b


x = tf.constant(np.array([[1, 1], [2, 2]]), dtype=tf.float32)
print(x)
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
