import os

from PIL import Image
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, AveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model

import numpy as np
from train_model.vocab import Vocab
from train_model.loaddata import load_data, ResizeImage

from train_model.vocab import Vocab
vocab = Vocab()
ocr_shape = (32, 120, 3) # height, width, channels
nb_classes = vocab.size
inputs = Input(shape = ocr_shape, name = "inputs")
conv1 = Convolution2D(32, 5, 5, name = "conv1")(inputs)
relu1 = Activation('relu', name="relu1")(conv1)
conv2 = Convolution2D(64, 5, 5, name = "conv2")(relu1)
relu2 = Activation('relu', name="relu2")(conv2)
pool2 = MaxPooling2D(pool_size=(2,2), border_mode='same', name="pool1")(relu2)
conv3 = Convolution2D(128, 3, 3, name = "conv3")(pool2)
relu3 = Activation('relu', name="relu3")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), border_mode='same', name="pool2")(relu3)
relu4 = Activation('relu', name="relu4")(pool3)
conv4 = Convolution2D(128, 3, 3, name="conv4")(relu4)
relu5 = Activation('relu', name="relu5")(conv4)
pool4 = AveragePooling2D(pool_size=(2,2), name="pool4")(relu5)
fl = Flatten()(pool4)
fc1 = Dense(nb_classes, name="fc1")(fl)
drop = Dropout(0.20, name = "dropout1")(fc1)
fc21= Dense(nb_classes, name="fc21", activation="softmax")(drop)
fc22= Dense(nb_classes, name="fc22", activation="softmax")(drop)
fc23= Dense(nb_classes, name="fc23", activation="softmax")(drop)
fc24= Dense(nb_classes, name="fc24", activation="softmax")(drop)
merged = merge([fc21, fc22, fc23, fc24], mode = 'concat', name = "merged")
model = Model(input = inputs, output = merged)
model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

model.load_weights("./model/weights-19.hdf5")

path = '/Users/diamond/PycharmProjects/TextRecognitionDataGenerator/TextRecognitionDataGenerator/out/'
imgs = os.listdir(path)

num = len(imgs)
i = 0
print(num)
for t in range(num):

    data = np.empty((1, 32, 120, 3))
    label = np.empty((1, 4 * vocab.size))
    vocab = Vocab()
    img = Image.open(path + imgs[t])
    img = ResizeImage(img, 120, 32)
    if imgs[t].split("_")[0] == "":
        continue
    # print(imgs[t].split("_")[0])
    text = imgs[t].split("_")[0].replace(" ", "")

    if text.__len__() != 4:
        continue
    arr = np.array(img)
    data[0, :, :, :] = arr / 256
    # print(text)
    k = vocab.text_to_one_hot(text)
    Y_pred = model.predict(data, 1, 1)
    Pred_text = Vocab().one_hot_to_text(Y_pred[0])
    if Pred_text.lower() != text.lower():
        # print(text,"----------------",Pred_text)
        i=i+1
    else:
        print(text, "----------------", Pred_text)
print(num-i)
print((1-i*1.0/num)*100,'%')