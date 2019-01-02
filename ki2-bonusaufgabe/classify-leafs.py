import os
from PIL import Image

traindatapath = './Train/'
testdatapath = './Test/'


def readimagedata(path):
    train_labels = []
    train_images = []

    for category in os.listdir(path):
        categorypath = path + category + '/'
        for img in os.listdir(categorypath):
            im = Image.open(categorypath + img)
            train_images.append(list(im.getdata()))
            train_labels.append(category)

    return [train_labels, train_images]


(train_labels, train_images) = readimagedata(traindatapath)
(test_labels, test_images) = readimagedata(testdatapath)


print('Train labels: ', train_labels)
print('Images:', len(train_images))