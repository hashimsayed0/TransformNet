from keras.datasets import cifar10
import numpy as np
import os
import random
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses, layers, models, metrics, Model
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import svm, metrics
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import torch
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from skimage import transform
import skimage
from keras.utils import to_categorical
import argparse


class Dataset:
    def __init__(self, transformType='all', transformNum=5):
        
        self.transformType = transformType # 'rotation' or  'all'
        self.transformNum = transformNum # 5 or 10 for transformation; 2,4,8 for rotation

        # load dataset
        (self.trainX, self.trainy), (self.testX, self.testy) = cifar10.load_data()

        self.trainy_binary = to_categorical(self.trainy)
        self.testy_binary = to_categorical(self.testy)

        self.ssltrainX = np.concatenate((self.trainX, self.testX), axis=0)
        self.ssltrainy = np.concatenate((self.trainy, self.testy), axis=0)

        self.ptrainX = np.zeros((self.ssltrainX.shape[0]*self.transformNum, self.ssltrainX.shape[1], self.ssltrainX.shape[2], self.ssltrainX.shape[3]))
        
    
    
    def downstream_data(self):
        return (self.trainX, self.trainy_binary), (self.testX, self.testy_binary)
    
    
    def preprocessing_rotation(self):
        
        ptrainy = np.zeros((self.ssltrainy.shape[0]*self.transformNum, self.ssltrainy.shape[1]))
        for i in range(self.ssltrainX.shape[0]):
            for j in range(self.transformNum):
                ptrainy[i * self.transformNum + j] = j
            
            self.ptrainX[i * self.transformNum] = self.ssltrainX[i]
            
            if self.transformNum == 2:
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 180)
            
            if self.transformNum == 4:
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 90)
                self.ptrainX[i * self.transformNum + 2] = transform.rotate(self.ssltrainX[i], 180)
                self.ptrainX[i * self.transformNum + 3] = transform.rotate(self.ssltrainX[i], 270)

            if self.transformNum == 8:
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 45)
                self.ptrainX[i * self.transformNum + 2] = transform.rotate(self.ssltrainX[i], 90)
                self.ptrainX[i * self.transformNum + 3] = transform.rotate(self.ssltrainX[i], 135)
                self.ptrainX[i * self.transformNum + 4] = transform.rotate(self.ssltrainX[i], 180)
                self.ptrainX[i * self.transformNum + 5] = transform.rotate(self.ssltrainX[i], 225)
                self.ptrainX[i * self.transformNum + 6] = transform.rotate(self.ssltrainX[i], 270)
                self.ptrainX[i * self.transformNum + 7] = transform.rotate(self.ssltrainX[i], 315)
        
        ptrainy = to_categorical(ptrainy)
        return (self.ptrainX, ptrainy)    
    
    def preprocessing_transform1(self):
        ptrainy = np.zeros((self.ssltrainy.shape[0]*self.transformNum, self.ssltrainy.shape[1]))
        if self.transformNum == 5:
            for i in range(self.ssltrainX.shape[0]):
                for j in range(self.transformNum):
                    ptrainy[i * self.transformNum + j] = j
                    
                self.ptrainX[i * self.transformNum] = self.ssltrainX[i]
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 180)
                self.ptrainX[i * self.transformNum + 2] = transform.warp(self.ssltrainX[i], transform.AffineTransform(scale=0.7))
                self.ptrainX[i * self.transformNum + 3] = transform.warp(self.ssltrainX[i], transform.AffineTransform(translation=(2,0)))
                self.ptrainX[i * self.transformNum + 4] = transform.warp(self.ssltrainX[i], transform.AffineTransform(shear=0.3))
        
        if self.transformNum == 10:
            for i in range(self.ssltrainX.shape[0]):
                for j in range(self.transformNum):
                    ptrainy[i * self.transformNum + j] = j
                    
                self.ptrainX[i * self.transformNum] = self.ssltrainX[i]
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 180)
                self.ptrainX[i * self.transformNum + 2] = transform.warp(self.ssltrainX[i], transform.AffineTransform(scale=0.7))
                self.ptrainX[i * self.transformNum + 3] = transform.warp(self.ssltrainX[i], transform.AffineTransform(scale=1.3))
                self.ptrainX[i * self.transformNum + 4] = transform.warp(self.ssltrainX[i], transform.AffineTransform(translation=(2,0)))
                self.ptrainX[i * self.transformNum + 5] = transform.warp(self.ssltrainX[i], transform.AffineTransform(translation=(-2,0)))
                self.ptrainX[i * self.transformNum + 6] = transform.warp(self.ssltrainX[i], transform.AffineTransform(translation=(0,2)))
                self.ptrainX[i * self.transformNum + 7] = transform.warp(self.ssltrainX[i], transform.AffineTransform(translation=(0,-2)))
                self.ptrainX[i * self.transformNum + 8] = transform.warp(self.ssltrainX[i], transform.AffineTransform(shear=0.3))
                self.ptrainX[i * self.transformNum + 9] = transform.warp(self.ssltrainX[i], transform.AffineTransform(shear=-0.3))

        ptrainy = to_categorical(ptrainy)
        return (self.ptrainX, ptrainy)

    def preprocessing_transform2(self):
        
        ptrainy = np.zeros((self.ssltrainy.shape[0]*self.transformNum, self.transformNum))
        trans_hyp = [0, 180, 0.7, 1.3, 0.3, -0.3, 0, 2, -2]
        translation_hyp = []
        for i in range(self.trainX.shape[0]):
            for j in range(self.transformNum):
                rot_index = np.random.randint(2)
                scale_index = np.random.randint(2,4)
                shear_index = np.random.randint(4,6)
                transh_index = np.random.randint(6,9)
                transw_index = np.random.randint(6,9)
                #6 - hp, 7 - hn, 8 - vp, 9 - vn
                trans_indices = []
                if transh_index == 7:
                    trans_indices.append(6)
                elif transh_index == 8:
                    trans_indices.append(7)
                
                if transw_index == 7:
                    trans_indices.append(8)
                elif transw_index == 8:
                    trans_indices.append(9)
                
                self.ptrainX[i * self.transformNum + j] = transform.rotate(self.ssltrainX[i], trans_hyp[rot_index])
                tform  = transform.AffineTransform(scale=trans_hyp[scale_index], translation = (trans_hyp[transw_index], trans_hyp[transh_index]), shear=trans_hyp[shear_index])
                self.ptrainX[i * self.transformNum + j] = transform.warp(self.ssltrainX[i], tform)
                
                ptrainy[i * self.transformNum + j, (rot_index, scale_index, shear_index)] = 1
                ptrainy[i * self.transformNum + j, trans_indices] = 1
        
        return (self.ptrainX, ptrainy)