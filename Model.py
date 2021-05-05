#import libraries
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
from tensorflow.keras.applications.vgg16 import VGG16
from skimage import transform
import skimage
from keras.utils import to_categorical
import argparse

class Model:
    def __init__(self, transformType= 'all', transformNum = 5, saved_name = 'vgg16', selfsupervised_epochs = 2, selfsupervised_batch_size = 128, supervised_epochs = 1, supervised_batch_size = 32, feature_layer_trained = 'block5_pool', feature_layer = 'block2_pool'):
        """
        Instantiates hyperparameters passed on when the object is created. All arguments are optional, as they all have a default setting.
        
        Arguments
            transformType: type of transformation to be applied ('rotation' or 'all')
            transformNum: number of transformations to be applied (2, 4, 8 for rotation; 5 or 10 for all transformations)
            saved_name: model to be used as backbone for pretexr and downstream tasks.
            selfsupervised_epochs: no of epochs to train pretext task for.
            selfsupervised_batch_size: batch size for pretext task training.
            supervised_epochs: no of epochs to train downstream task for.
            supervised_batch_size: batch size for downstream task training.
            feature_layer_trained: name of the layer in the given model until which the pretext task should be trained. 3 fully connected layers will be added on top of this layer for pretext task.
            feature_layer: name of the layer in the given model until which the model trained for pretext task would be extracted and trained for downstream task. 3 fully connected layers will be added on top of this layer for downstream task.
        """
        
        self.transformType = transformType
        self.transformNum = transformNum
        self.saved_name = saved_name
        self.selfsupervised_epochs = selfsupervised_epochs
        self.selfsupervised_batch_size = selfsupervised_batch_size
        self.supervised_epochs = supervised_epochs
        self.supervised_batch_size = supervised_batch_size
        self.feature_layer_trained = feature_layer_trained
        self.feature_layer = feature_layer


        #rotation prediction task: multi-class single label classifciation and hence the following settings
        if transformType=='rotation':
            self.activation = 'softmax'
            self.loss = 'categorical_crossentropy'
        #transformation prediction task (with preprocessing method 2): multi-class multi-label classification and hence the following settings
        else:
            self.activation = 'sigmoid'
            self.loss = 'binary_crossentropy'

    
    def get_conv_model(self):
        """
        Builds a convolutional model for pretext task

        Returns:
            model: built Keras model
        """

        #VGG16 is the base model on top of which 3 fully connected layers are added for pretext task
        base = VGG16(include_top=False, weights=None, input_shape=(32, 32, 3))

        #Retrieves the layer until which pretext task should be trained.
        l = base.get_layer(self.feature_layer_trained).output
        
        #Adds 3 fully connected layers along with batchnorm and dropout.
        l = layers.Flatten()(l)
        l = layers.BatchNormalization()(l)
        l = layers.Dropout(0.5)(l)
        l = layers.Dense(200, kernel_regularizer=regularizers.l1_l2(l1 = 0.0001, l2 = 0.0005), kernel_initializer='he_uniform')(l)
        l = layers.BatchNormalization()(l)
        l = layers.Activation('relu')(l)
        l = layers.Dropout(0.5)(l)
        l = layers.Dense(200, kernel_regularizer=regularizers.l1_l2(l1 = 0.0001, l2 = 0.0005), kernel_initializer='he_uniform')(l)
        l = layers.BatchNormalization()(l)
        l = layers.Activation('relu')(l)
        l = layers.Dropout(0.5)(l)
        l = layers.Dense(self.transformNum, kernel_regularizer=regularizers.l1_l2(l1 = 0.00005, l2 = 0.0001), activation = self.activation)(l)
        
        return tf.keras.Model(inputs = base.input, outputs = l)
    
    
    def lr_schedule(self, epoch, lr):
        """
        Schedules learning rate based on epochs during training. Used as a callback function
        
        Arguments:
            epoch: current epoch
            lr: current learning rate
        
        Returns:
            lr: Updated learning rate
        """
        if epoch == 50:
            return lr / 5.
        if epoch == 40:
            return lr / 5.
        if epoch == 30:
            return lr / 5.
    
        return lr

    
    def train_feat(self, ptrainX, ptrainy):
        """
        Trains the model for pretext task.

        Arguments:
            ptrainX: preprocessed training data for pretext task
            ptrainy: preprocessed psuedo labels for pretext task

        Returns:
            log: training log
        """
        #Saving training log to csv file
        csv_logger = CSVLogger('experiments/pretext_log', append=True, separator=';')
        
        #Creating model
        model = self.get_conv_model()

        #Printing model summary
        model.summary()

        #Compiling model with optimizer, loss, and evaluation metric
        model.compile(optimizer = optimizers.RMSprop(), loss = self.loss, metrics=['accuracy'])

        #Training model and saving log
        log = model.fit(ptrainX, ptrainy, 
                        epochs = self.selfsupervised_epochs, batch_size = self.selfsupervised_batch_size, 
                        shuffle = True, callbacks = [LearningRateScheduler(self.lr_schedule), csv_logger])
        
        #Saving model to use later to train on downstream task
        model.save('experiments/pretext')

        return log
    
    
    def get_cls_model(self):
        """
        Builds the model for downstream task

        Returns:
            model: built Keras model
        """
        
        #Loading model previously trained for pretext task
        model = tf.keras.models.load_model('experiments/pretext')

        #Retrieving the layer until which downstream task should be trained
        l = model.get_layer(self.feature_layer).output

        #Adding 3 fully connected layers with batchnorm and dropout
        l = layers.Flatten()(l)
        l = layers.BatchNormalization()(l)
        l = layers.Dropout(0.5)(l)
        l = layers.Dense(200, kernel_regularizer=regularizers.l1_l2(l1 = 0.0001, l2 = 0.0005), kernel_initializer='he_uniform')(l)
        l = layers.BatchNormalization()(l)
        l = layers.Activation('relu')(l)
        l = layers.Dropout(0.5)(l)
        l = layers.Dense(200, kernel_regularizer=regularizers.l1_l2(l1 = 0.0001, l2 = 0.0005), kernel_initializer='he_uniform')(l)
        l = layers.BatchNormalization()(l)
        l = layers.Activation('relu')(l)
        l = layers.Dropout(0.5)(l)
        l = layers.Dense(10, kernel_regularizer=regularizers.l1_l2(l1 = 0.0001, l2 = 0.0005), activation = 'softmax')(l)
        
        return tf.keras.Model(inputs = model.input, outputs = l)
    
    
    
    def train_cls(self, trainX, trainy_binary, testX, testy_binary, train_features = True):
        """
        Trains the model for downstream task in freezed or unfreezed setting. Freezed: layers trained in pretext are kept freezed and not trained for downstream task.​ Unfreezed: the layers trained in the pretext are retrained in downstream task.​ 

        Arguments:
            trainX: training data 
            trainy_binary: one-hot encoded training labels
            testX: testing data
            testy_binary: one-hot encoded testing labels
            train_features: set to True for unfreezed setting and False for freezed setting

        Returns:
            log: training log
        """
        #Based on setting, save csv file with training log
        if train_features:
            csv_logger = CSVLogger('experiments/downstream_unfreezed_log', append=True, separator=';')
        else:
            csv_logger = CSVLogger('experiments/downstream_freezed_log', append=True, separator=';')
        
        #build model
        cls_model = self.get_cls_model()
        
        #print model summary
        cls_model.summary()
        
        #For freezed setting, all the layers until the last 12 layers (3 fully connected layers with batchnorm, activations and dropout) are kept freezed and NOT trained.
        if not train_features:
            for l in cls_model.layers[:-12]:
                l.trainable = False
        
        #Create datagenerator for potential data augmentation
        datagen = ImageDataGenerator()        

        #Compile model with optimizer, loss, and evaluation metric
        cls_model.compile(optimizer = optimizers.RMSprop(), loss = 'categorical_crossentropy', metrics=['accuracy'])
        
        #Train the model and save log
        cls_log = cls_model.fit(datagen.flow(trainX, trainy_binary, batch_size=self.supervised_batch_size),
                                epochs = self.supervised_epochs, shuffle = True,
                                callbacks = [LearningRateScheduler(self.lr_schedule), csv_logger],
                                validation_data = (testX, 
                                                testy_binary))
        
        #Save models with relevant names
        if train_features:
            cls_model.save('experiments/downstream_unfreezed')
        else:
            cls_model.save('experiments/downstream_freezed')
        
        return cls_log


    