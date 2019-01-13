import os
import random
# from PIL import Image
import numpy as np
from skimage import io, color
from skimage.transform import resize

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


traindatapath = './Train/'
testdatapath = './Test/'


def readimagedata(path):
    train_labels = []
    train_images = []

    for category in os.listdir(path):
        categorypath = path + category + '/'
        for img in os.listdir(categorypath):
            im = resize(color.rgb2gray(io.imread(categorypath + img)), (100, 100))
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

(train_labels, train_images) = readimagedata(traindatapath)
(test_labels, test_images) = readimagedata(testdatapath)

shuffledata(train_labels, train_images)
shuffledata(test_labels, test_images)

train_images = train_images / 255
test_images = test_images / 255

# print(type(train_images))
# print(train_images)
# print(type(train_images[0]))
# print(train_images[0])
# print(type(train_images[0][0]))
# print(train_images[0][0])

# print('Train labels: ', train_labels)
# print('Images:', len(train_images))


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(33, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=100)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
