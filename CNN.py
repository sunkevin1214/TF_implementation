from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def create_W(shape):
    W = tf.truncated_normal(shape=shape, stddev=1.0, dtype=tf.float32)
    return tf.Variable(W)
def create_b(shape):
    b = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(b)

def conv_2d(W, x):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def pool_max(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# process the data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

#convolution and pooling
conv1_W = create_W([5, 5, 1, 32])
conv1_b = create_b([32])
conv1_h = tf.nn.relu(pool_max(conv_2d(conv1_W, tf.reshape(X, shape=[-1, 28, 28, 1])) + conv1_b))
#convolution and pooling
conv2_W = create_W([5, 5, 32, 64])
conv2_b = create_b([64])
conv2_h = tf.nn.relu(pool_max(conv_2d(conv2_W, conv1_h) + conv2_b))
#full connection layer
fc1_W = create_W([7*7*64, 1024])
fc1_b = create_b([1024])
fc1_h = tf.nn.relu(tf.matmul(tf.reshape(conv2_h, shape=[-1, 7*7*64]), fc1_W)+fc1_b)
# dropout
drop_prob = tf.placeholder(dtype=tf.float32)
drop_h = tf.nn.dropout(fc1_h, keep_prob=drop_prob)
#full connection layer
fc2_W = create_W([1024, 10])
fc2_b = create_b([10])
fc2_h = tf.matmul(fc1_h, fc2_W) + fc2_b




# softmax
loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=fc2_h)
equal_number = tf.equal(tf.argmax(fc2_h, 1), tf.argmax(Y, 1))



accuracy = tf.reduce_mean(tf.cast(equal_number, tf.float32))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            print('train_step:%d, train accuracy:%g'% (i, accuracy.eval(feed_dict={X:batch[0], Y:batch[1], drop_prob:1.0})))
        train_step.run(feed_dict={X:batch[0], Y:batch[1], drop_prob:0.5})
    print('test accuracy:%g'% (accuracy.eval(feed_dict={X:mnist.test.images, Y:mnist.test.labels, drop_prob:1.0})))



