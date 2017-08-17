import tensorflow as tf
import numpy as np

data =np.asarray([[
    [
    [1,2,3], [1,2,4], [1,2,5], [1,2,6], [1,2,7]

    ],
    [
    [2,2,3], [2,2,4], [2,2,5], [2,2,6], [2,2,7]


    ],
    [
    [3,2,3], [3,2,4], [3,2,5], [3,2,6], [3,2,7]



    ],
    [

    [4,2,3], [4,2,4], [4,2,5], [4,2,6], [4,2,7]


    ],
    [
    [5,2,3], [5,2,4], [5,2,5], [5,2,6], [5,2,7]



    ]

                  ]])
################################################
print(data.shape)

X = tf.placeholder(shape=[None, 5, 5, 3], dtype=tf.float32)
Y = tf.reshape(X, shape=[-1, 5*5, 3])

flag = np.zeros(shape=(1, 25, 3))
flag[:, [1,2,3,4], :] = 1
print(flag)

Y2 = tf.where(tf.cast(flag, tf.bool), Y, -Y)
Y1 = tf.reshape(Y2, shape=[-1, 5, 5, 3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Y2_value, Y_value, Y1_value = sess.run([Y2, Y, Y1], {X:data})
    print(Y_value, Y2_value, Y1_value)

############################################################
print(data.shape)

X = tf.placeholder(shape=[None, 5, 5, 3], dtype=tf.float32)
Y = tf.reshape(X, shape=[-1, 5*5, 3])
flag = np.zeros(shape=(1, 25, 3))
flag[:, [1,2,3,4], :] = 1
flag = np.reshape(flag, [1, 5, 5, 3])
Y2 = tf.where(tf.cast(flag, tf.bool), X, -X)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Y2_value, Y_value = sess.run([Y2, Y], {X:data})
    print(Y2_value)


