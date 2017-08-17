import tensorflow as tf
import numpy as np

# test if in tensorflow


input_size = 100
output_size = 10
num_hidden_layers = 5
hidden_layer_nurons = [input_size, 20, 30, 40, 50, 60, output_size]

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
output_each_hidden_layer = [X]
i = 1
with tf.name_scope('Layer{}'.format(i)):
    W = tf.truncated_normal(shape=[hidden_layer_nurons[i-1], hidden_layer_nurons[i]], name='W{}'.format(i))
    b = tf.constant(0.1, dtype=tf.float32, shape=[hidden_layer_nurons[i]], name='b{}'.format(i))
    Z = tf.nn.relu(tf.matmul(output_each_hidden_layer[i-1], W) + b, name='relu'.format(i))
    output_each_hidden_layer.append(Z)
i=2
with tf.name_scope('Layer{}'.format(i)):
    W = tf.truncated_normal(shape=[hidden_layer_nurons[i-1], hidden_layer_nurons[i]], name='W{}'.format(i))
    b = tf.constant(0.1, dtype=tf.float32, shape=[hidden_layer_nurons[i]], name='b{}'.format(i))
    Z = tf.nn.relu(tf.matmul(output_each_hidden_layer[i-1], W) + b, name='relu'.format(i))
    output_each_hidden_layer.append(Z)
i=3
with tf.name_scope('Layer{}'.format(i)):
    W = tf.truncated_normal(shape=[hidden_layer_nurons[i-1], hidden_layer_nurons[i]], name='W{}'.format(i))
    b = tf.constant(0.1, dtype=tf.float32, shape=[hidden_layer_nurons[i]], name='b{}'.format(i))
    Z = tf.nn.relu(tf.matmul(output_each_hidden_layer[i-1], W) + b, name='relu'.format(i))
    output_each_hidden_layer.append(Z)
i=4
with tf.name_scope('Layer{}'.format(i)):
    W = tf.truncated_normal(shape=[hidden_layer_nurons[i-1], hidden_layer_nurons[i]], name='W{}'.format(i))
    b = tf.constant(0.1, dtype=tf.float32, shape=[hidden_layer_nurons[i]], name='b{}'.format(i))
    Z = tf.nn.relu(tf.matmul(output_each_hidden_layer[i-1], W) + b, name='relu'.format(i))
    output_each_hidden_layer.append(Z)
i=5
with tf.name_scope('Layer{}'.format(i)):
    W = tf.truncated_normal(shape=[hidden_layer_nurons[i-1], hidden_layer_nurons[i]], name='W{}'.format(i))
    b = tf.constant(0.1, dtype=tf.float32, shape=[hidden_layer_nurons[i]], name='b{}'.format(i))
    Z = tf.nn.relu(tf.matmul(output_each_hidden_layer[i-1], W) + b, name='relu'.format(i))
    output_each_hidden_layer.append(Z)


input_data = np.random.random(size=[5, input_size])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('/tmp/test4', sess.graph)
    sess.run(output_each_hidden_layer[-1], {X:input_data})





