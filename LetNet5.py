from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import datetime

mnist = input_data.read_data_sets('MNIST_data', reshape=False, one_hot=True)
x_train, y_train = np.pad(mnist.train.images, ((0,0), (2,2), (2,2), (0,0)), mode='constant'), mnist.train.labels
x_validation, y_validation = np.pad(mnist.validation.images, ((0,0), (2,2), (2,2), (0,0)), mode='constant'), mnist.validation.labels
x_test, y_test = np.pad(mnist.test.images, ((0,0), (2,2), (2,2), (0,0)), mode='constant'), mnist.test.labels

print(x_train.shape, y_train.shape, x_validation.shape, x_test.shape)

def get_W(shape):
    return tf.Variable(tf.truncated_normal(shape))
def get_b(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))

def conv(W, x):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='Y')

conv1_W = get_W([5, 5, 1, 6])
conv1_b = get_b([6])
conv1_h = tf.nn.relu(conv(conv1_W, X) + conv1_b)
pool1_h = pool(conv1_h)

conv2_W = get_W([4, 4, 6, 16])
conv2_b = get_b([16])
conv2_h = tf.nn.relu(conv(conv2_W, pool1_h) + conv2_b)
pool2_h = pool(conv2_h)


fc1_W = get_W([5*5*16, 120])
fc1_b = get_b([120])
fc1_h = tf.nn.relu(tf.matmul(tf.reshape(pool2_h, shape=[-1, 5*5*16]), fc1_W) + fc1_b)

fc2_W = get_W([120, 84])
fc2_b = get_b([84])
fc2_h = tf.nn.relu(tf.matmul(fc1_h, fc2_W) + fc2_b)

fc3_W = get_W([84, 10])
fc3_b = get_b([10])
fc3_h = tf.matmul(fc2_h, fc3_W) + fc3_b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=fc3_h))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
accuracy_count = tf.equal(tf.argmax(fc3_h, 1), tf.argmax(Y, 1))
accuracy_rate = tf.reduce_mean(tf.cast(accuracy_count, tf.float32))

#begin
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
epochs = 10
batch = 128
starttime = datetime.datetime.now()
for i in range(epochs):
    idx = np.arange(0, len(x_train))
    np.random.shuffle(idx)
    x_train,y_train = x_train[idx],y_train[idx]
    num_example = len(x_train)
    for offset in range(0, num_example, batch):
        x_train_batch, y_train_batch = x_train[offset:offset+batch], y_train[offset:offset+batch]
        train_step.run(feed_dict={X:x_train_batch, Y:y_train_batch})
    validation_accuracy = accuracy_rate.eval(feed_dict={X:x_validation, Y:y_validation})
    print('EPOCH:%d, validation accuracy:%g' %(i+1, validation_accuracy))

print('test accuracy:%g'%(accuracy_rate.eval(feed_dict={X:x_test, Y:y_test})))
endtime = datetime.datetime.now()
print((endtime-starttime))

# 2min 36 sec, with test accuracy 93.53%

