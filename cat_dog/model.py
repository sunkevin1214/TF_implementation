import tensorflow as tf


def inference(train_data, train_label, batch_size, n_class):
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable('weights', shape=[3,3,3,16], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases =tf.get_variable('biases', shape=[16], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(train_data, weights, strides=[1,1,1,1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope("pool1") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name="norm1")

    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable('weights', shape=[3,3,16,16], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[16], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, scope.name)

    with tf.variable_scope('pool2') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name=scope.name)

    with tf.variable_scope('local3') as scope:
        input_data = tf.reshape(pool2, shape=[batch_size, -1])
        dim = input_data.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(value=0.1))
        local3 = tf.nn.relu(tf.matmul(input_data,weights) + biases, name=scope.name)
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', shape=[128, 128], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    with tf.variable_scope('softmax') as scope:
        weights = tf.get_variable('weights', shape=[128, n_class], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[n_class], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.nn.relu(tf.matmul(local4, weights) + biases)
    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="entropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
        return loss

def train(loss, lr):
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate =lr)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op =optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluate(logits, labels):
    with tf.variable_scope('evaluate') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
