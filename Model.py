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

class Model:
    def __init__(self, transformType= 'all', transformNum = 5, saved_name = 'vgg16', selfsupervised_epochs = 2, selfsupervised_batch_size = 128, supervised_epochs = 1, supervised_batch_size = 32, feature_layer_trained = 'block5_pool', feature_layer = 'block2_pool'):
        self.transformType = transformType # 'rotation' or 'all'
        self.transformNum = transformNum # 5 or 10 for transformation; 2,4,8 for rotation

        self.saved_name = saved_name #model name
        self.selfsupervised_epochs = selfsupervised_epochs
        self.selfsupervised_batch_size = selfsupervised_batch_size
        self.supervised_epochs = supervised_epochs
        self.supervised_batch_size = supervised_batch_size
        self.feature_layer_trained = feature_layer_trained
        self.feature_layer = feature_layer



        if transformType=='rotation':
            self.activation = 'softmax'
            self.loss = 'categorical_crossentropy'
        else:
            self.activation = 'sigmoid'
            self.loss = 'binary_crossentropy'

    
    def get_conv_model(self):
        base = VGG16(include_top=False, weights=None, input_shape=(32, 32, 3))
        l = base.get_layer(self.feature_layer_trained).output
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
        if epoch == 50:
            return lr / 5.
        if epoch == 40:
            return lr / 5.
        if epoch == 30:
            return lr / 5.
    
        return lr

    
    def train_feat(self, ptrainX, ptrainy):
        csv_logger = CSVLogger('pretext_log', append=True, separator=';')
        model = self.get_conv_model()
        model.summary()
        model.compile(optimizer = optimizers.RMSprop(), loss = self.loss, metrics=['accuracy'])
        log = model.fit(ptrainX, ptrainy, 
                        epochs = self.selfsupervised_epochs, batch_size = self.selfsupervised_batch_size, 
                        shuffle = True, callbacks = [LearningRateScheduler(self.lr_schedule), csv_logger])
        model.save('pretext')
        return log
    
    
    def get_cls_model(self):
        
        model = tf.keras.models.load_model('pretext')
        l = model.get_layer(self.feature_layer).output
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
        if train_features:
            csv_logger = CSVLogger('downstream_unfreezed_log', append=True, separator=';')
        else:
            csv_logger = CSVLogger('downstream_freezed_log', append=True, separator=';')
        cls_model = self.get_cls_model()
        cls_model.summary()
        
        if not train_features:
            for l in cls_model.layers[:-12]:
                l.trainable = False
        
        datagen = ImageDataGenerator()        
        cls_model.compile(optimizer = optimizers.RMSprop(), loss = 'categorical_crossentropy', metrics=['accuracy'])
        cls_log = cls_model.fit(datagen.flow(trainX, trainy_binary, batch_size=self.supervised_batch_size),
                                epochs = self.supervised_epochs, shuffle = True,
                                callbacks = [LearningRateScheduler(self.lr_schedule), csv_logger],
                                validation_data = (testX, 
                                                testy_binary))
        if train_features:
            cls_model.save('downstream_unfreezed')
        else:
            cls_model.save('downstream_freezed')
        return cls_log


    