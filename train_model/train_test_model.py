from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, AveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model

import numpy as np
from train_model.vocab import Vocab
from train_model.loaddata import load_data
import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter
def train():
    vocab = Vocab()
    batch_size = 128
    epochs = 20
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
    model.summary()
    # model.load_weights("./model/weights-10.hdf5")
    path = '/Users/diamond/PycharmProjects/TextRecognitionDataGenerator/TextRecognitionDataGenerator/out/'

    data,lable = load_data(path,vocab)
    checkpoint = ModelCheckpoint(r'./model/weights-{epoch:02d}.hdf5',
                                 save_weights_only=True)
    # model.fit_generator(MyCaptchaGenerator(batch_size,"traindata/"), 300, 10)
    model.fit(data,lable,batch_size=batch_size,epochs=epochs,callbacks=[checkpoint])

if __name__ == '__main__':
    train()