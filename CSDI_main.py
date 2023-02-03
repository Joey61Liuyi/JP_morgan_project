# -*- coding: utf-8 -*-
# @Time    : 2023/2/3 16:11
# @Author  : LIU YI

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math
from tensorflow.keras import datasets, layers, models, losses


def silu(x):
    return x * tf.keras.backend.sigmoid(x)


def create_data():
    train_data = np.load('stock_train.npy')
    test_data = np.load('stock_test.npy')


    train_data = tf.convert_to_tensor(train_data)
    test_data = tf.convert_to_tensor(test_data)

    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    test_data = tf.data.Dataset.from_tensor_slices(test_data)


    return train_data, test_data


def kaiming_normal(shape, dtype=tf.float32, partition_info=None):
    # Kaiming normal initialization
    return tf.random.normal(shape, mean=0., stddev=tf.math.rsqrt(2. / shape[0]))


# ====== config part ===========

num_channel = 64
num_steps = 50
diffusion_embedding_dim = 128
projection_dim = diffusion_embedding_dim
is_train = 1
set_t = -1


# =======initialization part=========

diff_model = models.Sequential()

inputs = tf.placeholder(tf.float32, shape = (2, 133, 234, 2))
B, K, L, inputdim = inputs.shape



# build embedding

tep = tf.range(0,num_steps,1)
tep = tf.expand_dims(
    tep, 1, name=None
)
tep = tf.cast(tep, dtype= tf.float32)
dim = diffusion_embedding_dim/2
frequencies = 10 ** tf.expand_dims((tf.range(dim) / (dim - 1) * 4.0), 0)
table = tep*frequencies
table = tf.concat([tf.math.sin(table),tf.math.cos(table)], axis = 1)


# input projection
inputs = tf.reshape(inputs, (B, K*L, inputdim))


conv = tf.keras.layers.Conv1D(
        num_channel,
        kernel_size=1,
        strides=1,
        activation='relu',
        kernel_initializer=kaiming_normal,
        input_shape=(2, 133, 234, 2)
    )


inputs = conv(inputs)
inputs = tf.reshape(inputs, (B, K, L, num_channel))

# diffusion_embedding

if is_train:
    steps = tf.random.uniform(shape=(B,), minval=0, maxval=num_steps, dtype=tf.int32)
else:
    steps = tf.cast((tf.ones(B)*set_t), tf.float32)


diffusion_embedding = tf.gather(table, indices=steps)
projection1 = tf.keras.layers.Dense(projection_dim, activation = silu)
diffusion_embedding = projection1(diffusion_embedding)
projection2 = tf.keras.layers.Dense(projection_dim, activation = silu)
diffusion_embedding = projection2(diffusion_embedding)

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
result = sess.run(diffusion_embedding)
print(result)
print('OK')

