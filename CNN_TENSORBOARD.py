from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def create_W(shape, name):
    W = tf.truncated_normal(shape=shape, stddev=1.0, dtype=tf.float32, name=name)
    return tf.Variable(W)
def create_b(shape, name):
    b = tf.constant(0.1, shape=shape, dtype=tf.float32, name=name)
    return tf.Variable(b)

def conv_2d(W, x, name='conv_ope'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
def pool_max(x, name='max_pool'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

with tf.name_scope("Input_Data"):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')

with tf.name_scope('conv1'):
    #convolution and pooling
    conv1_W = create_W([5, 5, 1, 32], name='weights')
    conv1_b = create_b([32], name='bias')
    conv1_h = tf.nn.relu(pool_max(conv_2d(conv1_W, tf.reshape(X, shape=[-1, 28, 28, 1])) + conv1_b), name='RELU')

with tf.name_scope('conv2'):
    #convolution and pooling
    conv2_W = create_W([5, 5, 32, 64], name='weights')
    conv2_b = create_b([64], name='bias')
    conv2_h = tf.nn.relu(pool_max(conv_2d(conv2_W, conv1_h) + conv2_b), name='RELU')

with tf.name_scope('FC1'):
    #full connection layer
    fc1_W = create_W([7*7*64, 1024], 'weights')
    fc1_b = create_b([1024], 'bias')
    fc1_h = tf.nn.relu(tf.matmul(tf.reshape(conv2_h, shape=[-1, 7*7*64]), fc1_W)+fc1_b, name='relu')
    # dropout
    drop_prob = tf.placeholder(dtype=tf.float32)
    drop_h = tf.nn.dropout(fc1_h, keep_prob=drop_prob, name='DROP')

with tf.name_scope('FC2'):
    #full connection layer
    fc2_W = create_W([1024, 10], 'weights')
    fc2_b = create_b([10], 'bias')
    fc2_h = tf.matmul(fc1_h, fc2_W) + fc2_b

with tf.name_scope('softmax'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=fc2_h, name='softmax'))

with tf.name_scope('train'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)

with tf.name_scope('evaluation'):
    equal_number = tf.equal(tf.argmax(fc2_h, 1), tf.argmax(Y, 1))
    correct = tf.cast(equal_number, tf.float32)
    accuracy = tf.reduce_mean(correct)

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)
merged_summary = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/cnn/train', graph=tf.get_default_graph())
test_writer = tf.summary.FileWriter('/tmp/cnn/test', graph=tf.get_default_graph())

# process the data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
MAX_EPOCH = 20
batch_size = 20


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(MAX_EPOCH):
        num_sample = len(mnist.train.images)
        for start_index in range(0, num_sample, batch_size):
            train_image_batch = mnist.train.images[start_index:start_index+batch_size,:]
            train_label_batch = mnist.train.labels[start_index:start_index+batch_size,:]
            summary_str, _, train_loss, train_acc = sess.run([merged_summary, train_op, loss, accuracy], {X: train_image_batch, Y:train_label_batch})
            if tf.train.global_step(sess, global_step) % 50 == 0:
                test_summary_str, test_acc = sess.run([merged_summary, accuracy], {X:mnist.test.images, Y:mnist.test.labels})
                train_writer.add_summary(summary_str, tf.train.global_step(sess, global_step))
                test_writer.add_summary(test_summary_str, tf.train.global_step(sess, global_step))
                print('epoch {}, step {}, loss {}, train_acc {}, test_acc {}'.format(epoch, tf.train.global_step(sess, global_step), train_loss, train_acc, test_acc))
    test_summary_str, test_acc = sess.run([merged_summary, accuracy], {X:mnist.test.images, Y:mnist.test.labels})
    test_writer.add_summary(test_summary_str, tf.train.global_step(sess, global_step))
    print('test acc: {}'.format(test_acc))











