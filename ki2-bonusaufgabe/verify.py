from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import operator
import numpy as np

# dimensions of images
img_width, img_height = 150, 150

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classify-leafs.h5")
loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# predicting images
img = image.load_img('Flavia/Test/020/2551.jpg', target_size=(img_width, img_height))
# print("=====================")
# print(img)
x = image.img_to_array(img)
# print("=====================")
# print(x)
x = np.expand_dims(x, axis=0)
# print("=====================")
# print(x)
x = x / 255.
# print("=====================")
# print(x)


# images = np.vstack([x])
# classes = loaded_model.predict(x)
# print(classes)

validation_data_dir = 'Flavia/Test'
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='categorical')
(img, img_class) = validation_generator.next()

index, value = max(enumerate(img_class[0]), key=operator.itemgetter(1))
print(img_class)
print('expected class:')
print(index)
classes = loaded_model.predict(img)
index, value = max(enumerate(classes[0]), key=operator.itemgetter(1))
print(classes)
print(index)


# classes = loaded_model.predict_generator(validation_generator)
# print(classes[0])