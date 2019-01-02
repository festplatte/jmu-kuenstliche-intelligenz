import os
from PIL import Image

traindatapath = './Train/'


def readTrainingData():
    train_labels = []
    train_images = []

    for category in os.listdir(traindatapath):
        categorypath = traindatapath + category + '/'
        for img in os.listdir(categorypath):
            im = Image.open(categorypath + img)
            train_images.append(list(im.getdata()))
            train_labels.append(category)
            # with open(categorypath + img, 'rb') as imgfile:
            #     imgdata = imgfile.read()
            #     train_images.append(imgdata)
            #     train_labels.append(category)
            break
        break

    return [train_labels, train_images]


result = readTrainingData()

print(result)
