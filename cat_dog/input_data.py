import numpy as np
import os
import tensorflow as tf
import DL1.cat_dog.model as model

def get_files():
    dir_name = '/am/lido/home/yanan/eclipse-workspace/DeepLearning/DL1/cat_dog/data/train'
    cat_train_images = []
    cat_train_label = []
    dog_train_images = []
    dog_train_label = []
    file_list = os.listdir(dir_name)
    for pic in file_list:
        if pic.split('.')[0] =='dog':
            dog_train_images.append(dir_name+pic)
            dog_train_label.append(1)
        else:
            cat_train_images.append(dir_name+pic)
            cat_train_label.append(0)
    print('number of cat {}'.format(len(cat_train_images)))
    print('number of dog {}'.format(len(dog_train_images)))

    images = np.hstack((cat_train_images, dog_train_images))
    labels = np.hstack((cat_train_label, dog_train_label))
    train_data = np.array([images, labels]).transpose()
    np.random.shuffle(train_data)
    train_images = train_data[:,0]
    train_label = train_data[:,1]
    train_label = [int(i) for i in train_label]
    return train_images, train_label

def get_batch(batch_size, image_height, image_width, capacity):

    images, labels = get_files()
    images = tf.cast(images, tf.string)
    labels = tf.cast(labels, tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels])
    labels = input_queue[1]
    images_contents = tf.read_file(input_queue[0])
    images = tf.image.decode_image(images_contents, channels=3)

    images = tf.image.resize_image_with_crop_or_pad(images, image_height, image_width)
    images = tf.image.per_image_standardization(images)
    images.set_shape([image_height, image_width, 3])
    image_batch, label_batch = tf.train.batch([images, labels], batch_size=batch_size, num_threads=36, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch

if __name__ == "__main__":
    n_class=2
    batch_size = 16
    capacity = 64
    image_height = 208
    image_width = 208
    MAX_STEP = 15000
    image_batch, label_batch = get_batch(batch_size, image_height, image_width, capacity)
    logits = model.inference(image_batch, label_batch, batch_size, n_class)
    loss = model.losses(logits, label_batch)
    train_op = model.train(loss, lr=0.0001)
    accuracy = model.evaluate(logits, label_batch)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for ite in range(MAX_STEP):
                if coord.should_stop():
                    break;
                [_, train_loss, train_accuracy] = sess.run([train_op, loss, accuracy])
                if ite % 50 ==0:
                    print('Step:{}, train loss:{}, train_accuracy:{}'.format(ite, train_loss, train_accuracy))
                if (ite + 1) == MAX_STEP:
                    print('Step:{}, train loss:{}, train_accuracy:{}'.format(ite, train_loss, train_accuracy))
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()
        coord.join(threads)



