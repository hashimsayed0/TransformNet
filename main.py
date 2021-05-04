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
from Dataset import Dataset
from Model import Model

def plot_training(**kwargs):
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

transformType= 'rotation'
transformNum = 2

parser = argparse.ArgumentParser()
parser.add_argument("-t","--transformation_type", type=str, help="enter rotation for only rotation and enter all for all transformations")
parser.add_argument("-n", "--transformation_num", type=int, help="enter no of transformations: 2, 4 or 8 for rotations; 5 or 10 for all transformations")
args = parser.parse_args()

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

print('Transformation type: ' + transformType)
print('Transformation number: ' + str(transformNum))



dataset = Dataset(transformType=transformType, transformNum=transformNum)
if transformType == 'rotation':
    (ptrainX, ptrainy) = dataset.preprocessing_rotation()
else:
    (ptrainX, ptrainy) = dataset.preprocessing_transform1()

model = Model(transformType, transformNum)
pretext_log = model.train_feat(ptrainX, ptrainy)

(trainX, trainy_binary), (testX, testy_binary) = dataset.downstream_data()
downstream_unfreezed_log = model.train_cls(trainX, trainy_binary, testX, testy_binary, True)
downstream_freezed_log = model.train_cls(trainX, trainy_binary, testX, testy_binary, False)

