import os
from PIL import Image

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


traindatapath = './Train/'
testdatapath = './Test/'

# TODO: not working yet
def greyscaleimage(image):
    result = []

    for row in image:
        newrow = []
        for pixel in row:
            newpixel = pixel[0] + pixel[1] + pixel[2]
            newrow.append(newpixel / 765)
        result.append(newrow)
    return result

def readimagedata(path):
    train_labels = []
    train_images = []

    for category in os.listdir(path):
        categorypath = path + category + '/'
        for img in os.listdir(categorypath):
            im = Image.open(categorypath + img)
            train_images.append(greyscaleimage(list(im.getdata())))
            train_labels.append(category)

    return [train_labels, train_images]


(train_labels, train_images) = readimagedata(traindatapath)
(test_labels, test_images) = readimagedata(testdatapath)

print('Train labels: ', train_labels)
print('Images:', len(train_images))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
