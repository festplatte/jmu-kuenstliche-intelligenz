import os
import random
import numpy as np
from skimage import io, color
from skimage.transform import resize

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

TRAIN_DATA_PATH = './Flavia/Train/'
TEST_DATA_PATH = './Flavia/Test/'
IMG_SIZE = 50
NUM_CLASSES = 32
EPOCHS = 5
USE_GRAYSCALE = True

image_channels = 1 if USE_GRAYSCALE else 3
random.seed(100)

def readimagedata(path):
    train_labels = []
    train_images = []

    for category in os.listdir(path):
        categorypath = path + category + '/'
        for img in os.listdir(categorypath):
            im = io.imread(categorypath + img)
            if USE_GRAYSCALE:
                im = color.rgb2gray(im)
            im = resize(im, (IMG_SIZE, IMG_SIZE))
            train_images.append(im)
            train_labels.append(int(category) - 1)

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

# Reading and pre-format the data
(train_labels, train_images) = readimagedata(TRAIN_DATA_PATH)
(test_labels, test_images) = readimagedata(TEST_DATA_PATH)

shuffledata(train_labels, train_images)
shuffledata(test_labels, test_images)

train_images = train_images.reshape(train_images.shape[0], IMG_SIZE, IMG_SIZE, image_channels)
test_images = test_images.reshape(test_images.shape[0], IMG_SIZE, IMG_SIZE, image_channels)

# Builds the model
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation=tf.nn.relu, input_shape=(IMG_SIZE, IMG_SIZE, image_channels)),
    # keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
    keras.layers.Conv2D(128, (5, 5), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trains the model
model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels))

# Checks the accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
