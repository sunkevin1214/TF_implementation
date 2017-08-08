import tensorflow as tf
import numpy as np

'''
this file is to test the shuffle function in tensorflow
1: we give a list of index
2: shuffle them and test whether their orders are to be changed in each loop
'''

a = tf.constant(np.arange(0, 10))
rs = tf.random_shuffle(a)
sess = tf.InteractiveSession()
for _ in range(10):
    print(sess.run(rs))

''' it prints as below

[1 4 2 9 8 7 5 3 6 0]
[2 6 0 1 8 3 7 5 9 4]
[7 9 8 4 3 5 1 2 0 6]
[1 3 0 2 6 4 8 7 9 5]
[9 5 1 3 7 0 6 2 4 8]
[5 1 9 6 2 3 4 8 0 7]
[6 8 5 3 4 1 7 0 2 9]
[4 9 6 3 2 8 1 7 0 5]
[1 7 6 5 0 3 9 4 8 2]
[6 0 4 3 2 1 7 5 8 9]

'''

