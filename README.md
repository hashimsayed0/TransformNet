# TransformNet: Self-supervised learning by predicting geometric transformations

## Abstract
Deep neural networks need a big amount of training data, while in the real world there is a scarcity of data available for training purposes. To resolve this issue unsupervised methods are used for training with limited data. In this report, we describe the unsupervised semantic feature learning approach for recognition of the geometric transformation applied to the input data. The basic concept of our approach is that if someone is unaware of the objects in the images, he/she would not be able to quantitatively predict the geometric transformation that was applied to them. This self supervised  scheme is based on pretext task and the downstream task. The pretext classification task to quantify the geometric transformations should force the CNN to learn high-level salient features of objects useful for image classification. In our baseline model, we define image rotations by multiples of 90 degrees. The CNN trained on this pretext task will be used for the classification of images in the CIFAR-10 dataset as a downstream task. we run the baseline method using various models, including  ResNet, DenseNet, VGG-16, and NIN with a varied number of rotations in feature extracting and fine-tuning settings. In extension of this baseline model we experiment with transformations other than rotation in pretext task. We compare performance of selected models in various settings with different transformations applied to images,various data augmentation techniques as well as using different optimizers. %In the first settings we predict the tranformations applied to images in multiple settings i.e 5 and 10. We pick the best performing model in rotation prediction task and check its performance in downstream task with varying augmentation strengths in three rotation settings i.e 2,4 and 8.
This series of different type of experiments will help us demonstrate the recognition accuracy of our self-supervised model when applied to a downstream task of classification.

## Architecture
![architecture](https://github.com/hashimsayed0/TransformNet/blob/main/architecture.png)

## Requirements
All requirements are stated in the environment.yml file using which you can create a conda environment that contains all required packages.

## Steps to run our code
1. Clone this repository: `git clone https://github.com/hashimsayed0/TransformNet.git`
2. Change directory to this project folder: `cd TransformNet`
3. Create a directory for experiments: `mkdir experiments`
4. Create conda environment from environment.yml file: `conda env create -f environment.yml`
5. Activate the environment, mash: `conda activate mash`
6. You can now run the main.py file with 2 optional arguments, transformation_type (all or rotation) and transformation_num (2, 4, 8 for rotation, 5 or 10 for all transformations)
   - To run the project with default settings (2 rotations), do not provide any optional arguments: `python main.py`
   - To run the project with 5 all transformations for example, use: `python main.py -t all -n 5`
   - To run the project with 8 rotations for example, use: `python main.py -t rotation -n 8`

### Note
Models, training logs and plots will be saved in experiments folder after the program runs successfully

## Dataset
The code uses CIFAR-10 dataset from keras datasets. So you don't have to download the dataset yourself.

## Credits
We used code from [this repository](https://github.com/WaretleR/SelfSupervision). Besides, we wrote our own code for preprocessing, building and training more models, plotting varied data and experimenting with more hyperparameters.
