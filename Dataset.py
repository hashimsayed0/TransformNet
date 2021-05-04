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
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from skimage import transform
import skimage
from keras.utils import to_categorical
import argparse


class Dataset:
    def __init__(self, transformType='all', transformNum=5):
        """
        Initializes variables for dataset preprocessing
        
        Transformation types:-
            'rotation' (2, 4 or 8): only rotation
            'all' (5 or 10): rotation, shearing, scaling and translation
        """
        
        self.transformType = transformType # 'rotation' or  'all'
        self.transformNum = transformNum # 5 or 10 for all transformations; 2,4,8 for rotation

        # load dataset
        (self.trainX, self.trainy), (self.testX, self.testy) = cifar10.load_data()

        #convert labels to one hot encoding
        self.trainy_binary = to_categorical(self.trainy)
        self.testy_binary = to_categorical(self.testy)

        #Combine train and test data into train data for pretext task => Entire dataset is used for training
        self.ssltrainX = np.concatenate((self.trainX, self.testX), axis=0)
        self.ssltrainy = np.concatenate((self.trainy, self.testy), axis=0)

        #Initialize training data for pretext task: (60,000 x transformNum, 32. 32. 3)
        self.ptrainX = np.zeros((self.ssltrainX.shape[0]*self.transformNum, self.ssltrainX.shape[1], self.ssltrainX.shape[2], self.ssltrainX.shape[3]))
        
    
    
    def downstream_data(self):
        """
        Returns dataset for training in downstream task

        Returns:
            trainX: training data 
            trainy_binary: one-hot encoded training labels
            testX: testing data
            testy_binary: one-hot encoded testing labels
        """
        return (self.trainX, self.trainy_binary), (self.testX, self.testy_binary)
    
    
    def preprocessing_rotation(self):
        """
        Performs preprocessing for pretext task of rotation prediction

        Returns:
            ptrainX: training data for pretext task
            ptrainy: training labels for pretext task
        """

        #initialize labels for pretext task: (60,000 x transformNum. 1)
        ptrainy = np.zeros((self.ssltrainy.shape[0]*self.transformNum, self.ssltrainy.shape[1]))
        
        #loop through each image in the combined dataset and apply rotations to each image
        for i in range(self.ssltrainX.shape[0]):
            
            #set labels of each transformed image
            for j in range(self.transformNum):
                ptrainy[i * self.transformNum + j] = j
            
            #first transformed image: original unrotated image
            self.ptrainX[i * self.transformNum] = self.ssltrainX[i]
            
            #second copy is rotated 180 degrees for 2 rotations
            if self.transformNum == 2:
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 180)
            
            #3 more rotations for 4 rotations: 90, 180, 270
            if self.transformNum == 4:
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 90)
                self.ptrainX[i * self.transformNum + 2] = transform.rotate(self.ssltrainX[i], 180)
                self.ptrainX[i * self.transformNum + 3] = transform.rotate(self.ssltrainX[i], 270)

            #7 more rotations for 8 rotations: 45, 90, 135, 180, 225, 270, 315
            if self.transformNum == 8:
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 45)
                self.ptrainX[i * self.transformNum + 2] = transform.rotate(self.ssltrainX[i], 90)
                self.ptrainX[i * self.transformNum + 3] = transform.rotate(self.ssltrainX[i], 135)
                self.ptrainX[i * self.transformNum + 4] = transform.rotate(self.ssltrainX[i], 180)
                self.ptrainX[i * self.transformNum + 5] = transform.rotate(self.ssltrainX[i], 225)
                self.ptrainX[i * self.transformNum + 6] = transform.rotate(self.ssltrainX[i], 270)
                self.ptrainX[i * self.transformNum + 7] = transform.rotate(self.ssltrainX[i], 315)
        
        #one-hot encode labels
        ptrainy = to_categorical(ptrainy)

        return (self.ptrainX, ptrainy)    
    
    def preprocessing_transform1(self):
        """
        Performs first preprocessing method for pretext task of transformation prediction: 
        Apply each transformation one at a time, producing 5 or 10 transformed images with only one transformation in each image.

        Returns:
            ptrainX: training data for pretext task
            ptrainy: training labels for pretext task
        """
        
        #initialize labels for pretext task: (60,000 x transformNum. 1)
        ptrainy = np.zeros((self.ssltrainy.shape[0]*self.transformNum, self.ssltrainy.shape[1]))
        
        #preprocessing loop for 5 transformations
        if self.transformNum == 5:

            #loop through each image in the combined dataset and apply transformations separately to each image
            for i in range(self.ssltrainX.shape[0]):
                
                #set labels of each transformed image
                for j in range(self.transformNum):
                    ptrainy[i * self.transformNum + j] = j
                    
                #apply 5 transformations separately: original, rotation, scaling, translation, shearing
                self.ptrainX[i * self.transformNum] = self.ssltrainX[i]
                self.ptrainX[i * self.transformNum + 1] = transform.rotate(self.ssltrainX[i], 180)
                self.ptrainX[i * self.transformNum + 2] = transform.warp(self.ssltrainX[i], transform.AffineTransform(scale=0.7))
                self.ptrainX[i * self.transformNum + 3] = transform.warp(self.ssltrainX[i], transform.AffineTransform(translation=(2,0)))
                self.ptrainX[i * self.transformNum + 4] = transform.warp(self.ssltrainX[i], transform.AffineTransform(shear=0.3))
        
        
        #preprocessing loop for 5 transformations
        if self.transformNum == 10:

            #loop through each image in the combined dataset and apply transformations separately to each image
            for i in range(self.ssltrainX.shape[0]):

                #set labels of each transformed image
                for j in range(self.transformNum):
                    ptrainy[i * self.transformNum + j] = j
                    
                #apply 5 transformations separately: original, rotation, scaling (0.7 and 1.3), translation(right, left, up, down), shearing (right and left)
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
        
        #one-hot encode labels
        ptrainy = to_categorical(ptrainy)
        
        return (self.ptrainX, ptrainy)

    def preprocessing_transform2(self):
        """
        Performs second preprocessing method for pretext task of transformation prediction: 
        Apply 10 random transformations to each image.

        Returns:
            ptrainX: training data for pretext task
            ptrainy: training labels for pretext task
        """
        
        #initialize labels for pretext task: (60,000 x transformNum. transformNum)
        ptrainy = np.zeros((self.ssltrainy.shape[0]*self.transformNum, self.transformNum))
        
        #hyperparameters for transformations
        trans_hyp = [0, 180, 0.7, 1.3, 0.3, -0.3, 0, 2, -2]

        #loop through each image in the combined dataset and apply random transformations to each image
        for i in range(self.ssltrainX.shape[0]):
            
            #apply transformnum number of transformations to each image
            for j in range(self.transformNum):

                #generate random indices for each transformation to be chosen from "trans_hyp" list of transformation hyperparamters
                rot_index = np.random.randint(2)
                scale_index = np.random.randint(2,4)
                shear_index = np.random.randint(4,6)
                transh_index = np.random.randint(6,9)
                transw_index = np.random.randint(6,9)

                #indices for translation
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
                
                #apply random rotation based on rot_index: 0 or 180 degrees
                self.ptrainX[i * self.transformNum + j] = transform.rotate(self.ssltrainX[i], trans_hyp[rot_index])

                #apply affine transformations with random translation (0,2 or -2 in height and width), shearing (0.3 or -0.3) and scaling (0.7 or 1.3) based on random indices generated
                tform  = transform.AffineTransform(scale=trans_hyp[scale_index], translation = (trans_hyp[transw_index], trans_hyp[transh_index]), shear=trans_hyp[shear_index])
                self.ptrainX[i * self.transformNum + j] = transform.warp(self.ssltrainX[i], tform)
                
                # set labels for pretext task: assign 1 to all the indices where corresponding transformation has been applied.
                # each example could get multiple positive label corresponding to the transformations applied.
                # [rot=0, rot=180, scale=0.7, scale=1.3, shear=0.3, shear=-0.3, transl=up, transl=down, transl=right, transl=left]
                # [0, 1, 0, 1, 1, 0, 1, 0, 0, 1] => Rotated 180 degrees, scaled by 1.7, sheared by 0.3, translated towards up and right
                ptrainy[i * self.transformNum + j, (rot_index, scale_index, shear_index)] = 1
                ptrainy[i * self.transformNum + j, trans_indices] = 1
        
        return (self.ptrainX, ptrainy)