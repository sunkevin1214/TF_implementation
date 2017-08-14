import tensorflow as tf

with tf.name_scope('Input'):
    a = tf.placeholder(dtype=tf.float32, shape=[1], name='a')
    b = tf.placeholder(dtype=tf.float32, shape=[1], name='b')

with tf.name_scope('ADD'):
    c = a + b

with tf.Session() as sess:
    writer = tf.summary.FileWriter('/tmp/test1', sess.graph)
    c_value = sess.run(c, {a:[2.0], b:[3.0]})
    print(c_value)

# just to initialize and writer insatnce for only save the graph. Noted that, only the placeholder can display the name space, and constant cannot display this scope name