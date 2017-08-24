import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


def get_data():
    mnist = input_data.read_data_sets('../MNIST_data', reshape=False, one_hot=True)
    data, label = mnist.train.images[0:100, :,:,:], mnist.train.labels[0:100, :]
    return data,label

def train():
    batch_size = 100
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1], name='Input')
    Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10], name='TrueLabel')

    #the first layer
    with tf.variable_scope('layer1'):
        W = tf.get_variable(name='W', shape=[5,5,1, 5], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', dtype=tf.float32, initializer=tf.constant(0.1, shape=[5]))
        H1 = tf.nn.relu(tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')) +  b

    #the second layer
    with tf.variable_scope('layer2'):
        H2 = tf.nn.max_pool(H1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #the third layer
    with tf.variable_scope('layer3'):
        D= tf.reshape(H2, shape=[batch_size, -1])
        input_dim = D.get_shape()[1].value
        W = tf.get_variable('W', shape=[input_dim, 10], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('b', dtype=tf.float32, shape=[10], initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        H3 = tf.nn.relu(tf.matmul(D, W) + b)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=H3), name="loss_op")
    with tf.name_scope('train'):
        train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(H3,1), tf.argmax(Y, 1)), dtype=tf.float32), name="accuracy_op")

    train_data, train_label = get_data()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./g', sess.graph)
        for _ in range(20):
            sess.run(train_op, {X:train_data, Y:train_label})
        saver0 = tf.train.Saver()
        saver0.save(sess, './save/model')
        saver0.export_meta_graph('./save/model.meta')
        for _ in range(5):
            loss_str, accuracy_str = sess.run([loss, accuracy], {X:train_data, Y:train_label})
            print('loss:{}, accuracy:{}'.format(loss_str, accuracy_str))


def load():
    train_data, train_label = get_data()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./save/model.meta')
        new_saver.restore(sess, './save/model')
        graph = sess.graph
        X = graph.get_tensor_by_name("Input:0")
        Y = graph.get_tensor_by_name('TrueLabel:0')
        loss = graph.get_tensor_by_name('loss/loss_op:0')
        accuracy = graph.get_tensor_by_name('accuracy/accuracy_op:0')
        for _ in range(5):
            loss_str, accuracy_str = sess.run([loss, accuracy], {X:train_data, Y:train_label})
            print('loss:{}, accuracy:{}'.format(loss_str, accuracy_str))


#train()
load()



