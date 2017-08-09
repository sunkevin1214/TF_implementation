from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

a = [[[1],[2]], [[2],[1]]], [[[3],[4]], [[2],[3]]], [[[1],[3]], [[1],[1]]], [[[2], [1]], [[1],[2]]], [[[4], [5]], [[2], [3]]]
a = np.asarray(a)

X = tf.placeholder(tf.float32, shape=[None, 2, 2, 1])

idx =  tf.random_shuffle(tf.range(0, tf.shape(X)[0]))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i_ in range(10):
    print(sess.run(idx,{X:a}))
