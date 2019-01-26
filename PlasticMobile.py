#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:16:15 2019

@author: kevinphan
"""

#I have decided to use TFLearn which was built on top of tensorflow. 

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2  
from random import shuffle
from tqdm import tqdm
import glob
from PIL import Image
import shutil

os.chdir('/Users/kevinphan/Desktop/InterviewExam-master 3/ImageClassification')

#First, since the photos are not labelled, I am going to label them externally in order to create a one hot encoding environemnt. The data comes in 11 folders which each has images of each letter.
#Loop through the folders in our directory and rename accordingly. (A.1, B.5 so on and so forth) In folder 0 is A in folder 1 its B etc.

for Dir in os.listdir("."):
        if Dir == '0':
            for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "A." + str(i) + ".jpg")
        elif Dir == '1':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "B." + str(i) + ".jpg")
        elif Dir == '2':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "C." + str(i) + ".jpg")
        elif Dir == '3':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "D." + str(i) + ".jpg")
        elif Dir == '4':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "E." + str(i) + ".jpg")
        elif Dir == '5':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "F." + str(i) + ".jpg")
        elif Dir == '6':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "G." + str(i) + ".jpg")
        elif Dir == '7':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "H." + str(i) + ".jpg")
        elif Dir == '8':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "I." + str(i) + ".jpg")
        elif Dir == '9':
             for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "J." + str(i) + ".jpg")
        else:
            for i, filename in enumerate(os.listdir(str(Dir))):
                os.rename(Dir + "/" + filename, Dir + "/" + "K." + str(i) + ".jpg")

os.chdir('/Users/kevinphan/Desktop/InterviewExam-master 3')

#Now that the images are renamed, I am creating a folder to place all the labelled images into and that will act as our training / testing images.

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
createFolder('./TRAIN_/')
        
dst = '/Users/kevinphan/Desktop/InterviewExam-master 3/TRAIN_'
        
os.chdir('/Users/kevinphan/Desktop/InterviewExam-master 3/ImageClassification')

#I am no taking all the photos and placing it into this folder and this will be our training directory for the neural net. This is just a faster way of doing the manual work.

for dir in os.listdir("."):
    source_dir = '/Users/kevinphan/Desktop/InterviewExam-master 3/ImageClassification/{}'.format(dir)
    files = glob.iglob(os.path.join(source_dir, "*.jpg"))
    for file in files:
        shutil.copy2(file, dst) 
        
#______________________________________________________________________________________________________________________
        

TRAIN_DIR = '/Users/kevinphan/Desktop/InterviewExam-master 3/TRAIN_'

LR = 1e-3 #Learning rate of our model

MODEL_NAME = 'letters-{}-{}.model'.format(LR, '2conv-basic')

IMAGE_SIZE = 10 #The image sizes are all the same but are 16x11. I want perfect squares so lets set it to 10x10



#Defined a helper function to create a one hot enocding environment based on the word label. This will act as our matrix to create our 
#training data. According to our label which is defined by splitting the image name to isolate the letter, it will return the array with a one for the label that is has.
def label_img(img):
    word_label = img.split('.')[-2]
    if word_label == 'A': return [1,0,0,0,0,0,0,0,0,0,0]
    elif word_label == 'B': return [0,1,0,0,0,0,0,0,0,0,0]
    elif word_label == 'C': return [0,0,1,0,0,0,0,0,0,0,0]
    elif word_label == 'D': return [0,0,0,1,0,0,0,0,0,0,0]
    elif word_label == 'E': return [0,0,0,0,1,0,0,0,0,0,0]
    elif word_label == 'F': return [0,0,0,0,0,1,0,0,0,0,0]
    elif word_label == 'G': return [0,0,0,0,0,0,1,0,0,0,0]
    elif word_label == 'H': return [0,0,0,0,0,0,0,1,0,0,0]
    elif word_label == 'I': return [0,0,0,0,0,0,0,0,1,0,0]
    elif word_label == 'J': return [0,0,0,0,0,0,0,0,0,1,0]
    elif word_label == 'K': return [0,0,0,0,0,0,0,0,0,0,1]

#Created a helper function to transform the training images and their labels into numpy arrays. We take the image label, and the image
#and we append them into our training data array and shuffle them. First being the image and the second being the label. 

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)  
        img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
        np.expand_dims(img, axis=0)
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

train_data = create_train_data()

train = train_data[:-500]  #Creating our train and test sets.
test = train_data[-500:]


#resizing the arrays in order to feed them into the neural network. Here, -1 refers to the amount of files (batch size), the dimmensions and
#the last one refers to it being a gray scale image since all images are not coloured. This is where i inspected it the most for my error.

X = np.array([i[0] for i in train]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
Y = [i[1] for i in train] #Labels

test_x = np.array([i[0] for i in test]).reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
test_y = [i[1] for i in test]

#No we are ready to define the nueral network. We have used a CNN with 6 convolution layers. 

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#Defining our neural network

conNN = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')  #This is basically our placeholder function. Determining batch, dimmensions and channel.

conNN = conv_2d(conNN, 32, 5, activation='relu')
conNN = max_pool_2d(conNN, 5)

conNN = conv_2d(conNN, 64, 5, activation='relu')
conNN = max_pool_2d(conNN, 5)

conNN = conv_2d(conNN, 128, 5, activation='relu')
conNN = max_pool_2d(conNN, 5)

conNN = conv_2d(conNN, 64, 5, activation='relu')
conNN = max_pool_2d(conNN, 5)

conNN = conv_2d(conNN, 32, 5, activation='relu')
conNN = max_pool_2d(conNN, 5)

conNN = fully_connected(conNN, 1024, activation='relu') #Fully connected layer 
conNN = dropout(conNN, 0.8)

conNN = fully_connected(conNN, 11, activation='softmax') #For 11 classes ie A - K
conNN = regression(conNN, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(conNN, tensorboard_dir='log') #Here we define our model. 


#We will now train our model and test it against the validation set which is our test_x for images and test_y for labels. 

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#Aftermath
#once this code is ran. I get an error: Cannot feed value of shape (64, 10, 10, 1) for Tensor 'input/X:0', which has shape '(?, 50, 50, 1)'.
#I have checked my image parameters which i have reshaped to match the tensor input data but the error continues. Would love any input
#to improve the code by debuging the error. I defnitely think it is a broadcasting issue and is probably a small fix. 


