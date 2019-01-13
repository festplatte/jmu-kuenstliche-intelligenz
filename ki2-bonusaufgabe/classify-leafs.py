import os
import random
import numpy as np
from skimage import io, color
from skimage.transform import resize

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression


TRAIN_DATA_PATH = './Train/'
TEST_DATA_PATH = './Test/'
IMG_SIZE = 100
NUM_CLASSES = 33


def readimagedata(path):
    train_labels = []
    train_images = []

    for category in os.listdir(path):
        categorypath = path + category + '/'
        for img in os.listdir(categorypath):
            # im = resize(io.imread(categorypath + img), (IMG_SIZE, IMG_SIZE))
            im = resize(color.rgb2gray(io.imread(categorypath + img)), (IMG_SIZE, IMG_SIZE))
            # im = Image.open(categorypath + img).resize((50, 50), Image.ANTIALIAS).convert('L')
            # train_images.append(list(im.getdata()))
            train_images.append(im)
            train_labels.append(category)
        #     break
        # break

    # return [train_labels, train_images]
    return [np.array(train_labels), np.array(train_images)]

def shuffledata(labels, images):
    for i in range(0, len(labels)):
        j = random.randint(0, len(labels)-1)
        temp = labels[i]
        labels[i] = labels[j]
        labels[j] = temp
        temp = images[i]
        images[i] = images[j]
        images[j] = temp

(train_labels, train_images) = readimagedata(TRAIN_DATA_PATH)
(test_labels, test_images) = readimagedata(TEST_DATA_PATH)

shuffledata(train_labels, train_images)
shuffledata(test_labels, test_images)

# train_images = train_images / 255
# test_images = test_images / 255
train_images = train_images.reshape(train_images.shape[0], IMG_SIZE, IMG_SIZE, 1)
test_images = test_images.reshape(test_images.shape[0], IMG_SIZE, IMG_SIZE, 1)

# print(type(train_images))
# print(train_images)
# print(type(train_images[0]))
# print(train_images[0])
# print(type(train_images[0][0]))
# print(train_images[0][0])
# print(type(train_images[0][0][0]))
# print(train_images[0][50][50])

# print('Train labels: ', train_labels)
# print('Images:', len(train_images))


# tf.reset_default_graph()
# convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 128, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)
# convnet = fully_connected(convnet, 64, activation='softmax')
# convnet = regression(convnet, optimizer='adam', loss='categorical_crossentropy')
# model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
# model.fit(train_images, train_labels, n_epoch=10, validation_set=(test_images, test_labels), snapshot_step=500, show_metric=True, run_id='my_model')

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # keras.layers.Conv2D(128, 5, strides=(5, 5), activation=tf.nn.relu),
    # keras.layers.MaxPooling2D(pool_size=(5,5)),
    # keras.layers.Conv2D(64, 5, strides=(5, 5), activation=tf.nn.relu),
    # keras.layers.MaxPooling2D(pool_size=(5,5)),
    # keras.layers.Conv2D(32, 5, strides=(5, 5), activation=tf.nn.relu),
    # keras.layers.MaxPooling2D(pool_size=(5,5)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    # keras.layers.Dropout(0.8),
    keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# print('Should be label: ', test_images[0])
# print(model.predict([test_images[0]]))