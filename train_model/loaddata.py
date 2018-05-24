#coding:utf-8

import os
from PIL import Image
import numpy as np
from train_model.vocab import Vocab
#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]



def load_data(path, vocab=None):
    imgs = os.listdir(path)
    num = len(imgs)
    data = np.empty((num,32,120,3))
    label = np.empty((num,4*vocab.size))

    print(num)
    for t in range(num):
        vocab = Vocab()
        img = Image.open(path+imgs[t])
        img = ResizeImage(img,120,32)
        if imgs[t].split("_")[0]=="":
            continue
        # print(imgs[t].split("_")[0])
        text = imgs[t].split("_")[0].replace(" ","")

        if text.__len__()!=4:
            continue
        arr = np.array(img)
        data[t,:,:,:] = arr/256
        # print(text)
        k = vocab.text_to_one_hot(text)
        label[t] = k
    return data,label

def ResizeImage(img, width, height):
  out = img.resize((width, height),Image.ANTIALIAS) #resize image with high-quality
  return out


if __name__ == '__main__':
    path = '/Users/diamond/PycharmProjects/TextRecognitionDataGenerator/TextRecognitionDataGenerator/out/'
    vocab  =Vocab()
    print(load_data(path,vocab))
