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
from Dataset import Dataset
from Model import Model

def plot_training(**kwargs):
    """Plots the training log
    
    #Arguments

        name: plot title
        filename: name of file to be saved
        training_log: log output after training
    
    #Returns
        plot figure

    """
    plt.figure(figsize = (10, 10))
     
    for k, v in kwargs.items():
        if k != 'name' and k != 'filename':
            plt.plot(v, label=k)
                
    plt.grid(True)
    if 'name' in kwargs:
        plt.title(kwargs['name'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    if 'filename' in kwargs:
        plt.savefig(kwargs['filename'])

#Default training settings
transformType= 'rotation'
transformNum = 2

#Setting up argument parser to accept arguments from CLI for the purpose of settings for training
parser = argparse.ArgumentParser()
parser.add_argument("-t","--transformation_type", type=str, help="enter rotation for only rotation and enter all for all transformations")
parser.add_argument("-n", "--transformation_num", type=int, help="enter no of transformations: 2, 4 or 8 for rotations; 5 or 10 for all transformations")
args = parser.parse_args()

#Validating and saving arguments from CLI to variables
if ((args.transformation_type == 'rotation' and args.transformation_num in [2, 4, 8]) or (args.transformation_type == 'all' and args.transformation_num in [5, 10])):
    transformType = args.transformation_type
    transformNum = args.transformation_num

elif (args.transformation_type == 'all' and args.transformation_num is None):
    transformType= 'all'
    transformNum = 5

elif (args.transformation_type is None and args.transformation_num in [2, 4, 8]):
    transformType = 'rotation'
    transformNum = args.transformation_num

elif (args.transformation_type is None and args.transformation_num in [5, 10]):
    transformType = 'all'
    transformNum = args.transformation_num

elif (args.transformation_type is not None and args.transformation_num is not None):
    print("The entered arguments are not correct. Please rerun the program without arguments to train with default settings or use --help for more information on arguments.")
    quit()

#Printing the training settings
print('Transformation type: ' + transformType)
print('Transformation number: ' + str(transformNum))


#Creating dataset with chosen or default settings
dataset = Dataset(transformType=transformType, transformNum=transformNum)

#Calling relevant preprocessing methods
if transformType == 'rotation':
    (ptrainX, ptrainy) = dataset.preprocessing_rotation()
else:
    (ptrainX, ptrainy) = dataset.preprocessing_transform1()

#Creating model with chosen or default settings
model = Model(transformType, transformNum)

#Training for pretext task
pretext_log = model.train_feat(ptrainX, ptrainy)

#Retrieving datasets for downstream task
(trainX, trainy_binary), (testX, testy_binary) = dataset.downstream_data()

#Downstream task training with freezed and unfreezed settings
#Unfreezed: the layers trained in the pretext are retrained in downstream task.​
downstream_unfreezed_log = model.train_cls(trainX, trainy_binary, testX, testy_binary, True)

#Freezed: layers trained in pretext are kept freezed and not trained for downstream task.​
downstream_freezed_log = model.train_cls(trainX, trainy_binary, testX, testy_binary, False)

#Plotting training accuracy in downstream task
plot_training(name = 'Train accuracy (Downstream)',
                filename = 'train_plot',
                Unfreezed = downstream_unfreezed_log.history['accuracy'],
                Freezed = downstream_freezed_log.history['accuracy'])

#Plotting validation accuracy in downstream task
plot_training(name = 'Validation accuracy (Downstream)',
                filename = 'validation_plot',
                Unfreezed = downstream_unfreezed_log.history['val_accuracy'],
                Freezed = downstream_freezed_log.history['val_accuracy'])