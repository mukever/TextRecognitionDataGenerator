from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, AveragePooling2D, Reshape, GRU, K, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model

import numpy as np

import os
from PIL import Image, ImageFont, ImageDraw, ImageFilter

from train_model.vocab import Vocab

def load_data(path, vocab=None,conv_shape=None):
    imgs = os.listdir(path)
    num = len(imgs)
    data = np.zeros((num,32,90,3))
    label = np.zeros((num,4))
    print(num)
    for t in range(num):
        vocab = Vocab()
        img = Image.open(path+imgs[t])
        img = ResizeImage(img,90,32)
        if imgs[t].split("_")[0]=="":
            continue
        # print(imgs[t].split("_")[0])
        text = imgs[t].split("_")[0].replace(" ","")
        if text.__len__()!=4:
            continue
        arr = np.array(img)
        data[t,:,:,:] = arr
        # print(text)
        k = np.array(vocab.s_to_seq(text))
        label[t] = k
    print(data)
    print(label)
    return [data,label,np.ones(num)*int(conv_shape[1]-2),
               np.ones(num)*4], np.ones(num)

def ResizeImage(img, width, height):
  out = img.resize((width, height),Image.ANTIALIAS) #resize image with high-quality
  return out
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


from keras.callbacks import *


def train():
    rnn_size = 128
    vocab = Vocab()
    batch_size = 64
    n_len = 4
    epochs = 10
    ocr_shape = (32, 90, 3) # height, width, channels
    nb_classes = vocab.size
    inputs = Input(shape = ocr_shape, name = "inputs")
    conv1 = Convolution2D(32, 5, 5, name = "conv1")(inputs)
    relu1 = Activation('relu', name="relu1")(conv1)
    conv2 = Convolution2D(64, 5, 5, name = "conv2")(relu1)
    relu2 = Activation('relu', name="relu2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2), border_mode='same', name="pool2")(relu2)
    conv3 = Convolution2D(128, 3, 3, name = "conv3")(pool2)
    relu3 = Activation('relu', name="relu3")(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2), name="pool3")(relu3)

    conv_shape = pool3.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(pool3)

    x = Dense(32, activation='relu')(x)

    gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 init='he_normal', name='gru1_b')(x)
    gru1_merged = merge([gru_1, gru_1b], mode='sum')

    gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 init='he_normal', name='gru2_b')(gru1_merged)
    x = merge([gru_2, gru_2b], mode='concat')
    x = Dropout(0.25)(x)
    x = Dense(vocab.size, init='he_normal', activation='softmax')(x)
    base_model = Model(input=inputs, output=x)

    labels = Input(name='the_labels', shape=[4], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([x, labels, input_length, label_length])

    model = Model(input=[inputs, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    from keras.utils import plot_model
    plot_model(model, to_file='model_ctc.png')
    path = '/Users/diamond/PycharmProjects/TextRecognitionDataGenerator/TextRecognitionDataGenerator/out/'


    checkpoint = ModelCheckpoint(r'./model_ctc/weights-{epoch:02d}.hdf5',
                                 save_weights_only=True)

    model.fit(load_data(path,vocab,conv_shape),batch_size=320, nb_epoch=20,
                        callbacks=[checkpoint])

if __name__ == '__main__':
    train()