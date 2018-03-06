# -*- coding: utf-8 -*-
"""
Created on Sun May 21 01:59:42 2017

@author: Arpan
Description: Script to train simple ML models on camera frames. We use HOG features 
extracted from positive and negative frames. The models are saved on disk and used 
to extract cricket shots by detecting the starting frames as positive examples.

Refer: opencv/samples/python/digits.py  for the SVM usage details.
"""

import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
from sklearn.externals import joblib

# Server Param
#PROJECT_PREFIX = "/home/arpan/VisionWorkspace/shot_detection"
PROJECT_PREFIX = "/home/hadoop/VisionWorkspace/Cricket/scripts"
CAM1_FRAMES_DATA = "camera_models/cam1_frames"
CAM2_FRAMES_DATA = "camera_models/cam2_frames"
CLASS_N = 2

# read folder images paths and interpret labels from their names
# function returns the list of image paths and corresponding labels
def load_data(srcDataPath):
    img_list = list()
    labels_list = list()
    # read the images from the folder 
    files_list = sorted(os.listdir(srcDataPath))
    # save the images and labels in lists
    for path in files_list:
        img_list.append(cv2.imread(os.path.join(srcDataPath, path), 0))     #Grayscale
        if "pos" in path:       # positive example
            labels_list.append(1)
        else:                   # negative example
            labels_list.append(0)
    return img_list, labels_list

# Function to find the HOG descriptor
# Refer: mccormickml.com/2013/05/07/gradient-vectors
# mccormickml.com/2013/05/09/hog-person-detector-tutorial
# Also refer: HOG_test.py for how to set params
def preprocess_hog(images_list):
    # assumed that images have shape (h,w) = (360,640)
    samples = []
    hog = cv2.HOGDescriptor(os.path.join(PROJECT_PREFIX, "supporting_files/hog.xml"))
    # winStride, padding and locations are not needed
    for img in images_list:        
        #image = cv2.imread(imgPath, 0)      # taking only grascale image
        print("Image: "+str(img.shape))
        hist = hog.compute(img)
        
###############################################################################
        # calculate the Sobel gradients 
#        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)   # for each pixel
#        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#        mag, ang = cv2.cartToPolar(gx, gy)      # for each pixel, ang in radian
#        bin_n = 16
#        bin = np.int32(bin_n*ang/(2*np.pi))     # mapping ang to 16 bins (0-15)
#        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
#        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
#        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b,m in zip(bin_cells, mag_cells)]
#        hist = np.hstack(hists)
        
        # transform to Hellinger kernel
        # Refer digits.py in samples for details
#        eps = 1e-7
#        hist /= hist.sum() + eps
#        hist = np.sqrt(hist)
#        hist /= np.linalg.norm(hist) + eps
###############################################################################
        
        samples.append(hist)

    return np.float32(samples)

# Evaluate the trained model
def evaluate_model(model, samples, labels):
    resp = np.int32(model.predict(samples))
    print "Labels and corresponding predictions: "
    print zip(labels, resp)
    err = (labels!=resp).mean()
    print('error: %.2f %%' % (err*100))

    confusion = np.zeros((2,2), np.int32)
    for i,j in zip(labels, resp):
        confusion[i,j] += 1
#    # for multi-class classification (on MNIST 10-categories)
#    confusion = np.zeros((10,10), np.int32)
#    for i,j in zip(labels, resp):
#        confusion[i,j] += 1
    print("Confusion Matrix:")
    print(confusion)
    print()
    

# display images for validating the images and labels
def display_images(images, labels):
    # iterate over the images and display information
    for i in range(len(labels)):
        img = images[i]
        cv2.imshow("Image", img)
        print "Image "+str((i+1))+" : Label :: "+str(labels[i])
        waitTillEscPressed()
    cv2.destroyAllWindows()
    return

# helper functions
def waitTillEscPressed():
    while(True):
        if cv2.waitKey(10)==27:
            print("Esc Pressed")
            return
            

# Function to take input samples (frames) and train an SVM on HOG features of frames
# The frames dataset is created using the extract_frames_main.py file
def train_model(framesPath, shuffleSeed=123):
    # load the data, data is list of N ndarrays of 360x640 
    # labels is list of N binary values
    st_frames_data, labels = load_data(framesPath)
    # for shuffling data and labels need to be numpy arrays
    st_frames_data = np.array(st_frames_data)   # N x 360 x 640  (N, 360, 640)
    labels = np.array(labels)                   # N length vector (N,)
    
    # shuffle the data
    rand = np.random.RandomState(shuffleSeed)
    shuffle = rand.permutation(len(labels))
    print shuffle
    print(len(st_frames_data), labels)    
    
    # shuffling the data and the labels, w.r.t. first dimension of array
    st_frames_data, labels = st_frames_data[shuffle], labels[shuffle]
    #display_images(st_frames_data, labels)
    print(len(st_frames_data), labels)    
    
    # Features to be extracted from the samples
    # HOG : returns a list of 79380x1 vectors
    samples = preprocess_hog(list(st_frames_data))
    # check whether features are computed correctly
    
    # Divide into training and testing sets
    train_n = int(0.7*len(samples))
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])
        
    ###########################################################################
    # Reshape
    samples_train = samples_train.reshape((samples_train.shape[0], samples_train.shape[1]))
    samples_test = samples_test.reshape((samples_test.shape[0], samples_test.shape[1]))
    
    # Train a model    
    #model.train(samples_train, labels_train)    
    model = svm.SVC(kernel = 'linear')
    
    # Train a Random Forest model
    #model = RandomForestClassifier(max_depth=2, n_estimators=1000, random_state=1234)
    #clf.fit(df_train.loc[:, df_train.columns != 'Y'], df_train.loc[:,'Y'])
    model.fit(samples_train, labels_train)
    
    ###########################################################################
    # Evaluate the algorithm
    print "Evaluation on the test data:"
    evaluate_model(model, samples_test, labels_test)
    
    return model
    

if __name__=='__main__':
    
    # Train the model on 
    model1 = train_model(os.path.join(PROJECT_PREFIX, CAM1_FRAMES_DATA), 321)
    # save model to disk
    joblib.dump(model1, os.path.join(PROJECT_PREFIX, 
                                     "supporting_files/cam1_svm.pkl"))
    
    # Save the model to disk
    model2 = train_model(os.path.join(PROJECT_PREFIX, CAM2_FRAMES_DATA), 456)
    joblib.dump(model2, os.path.join(PROJECT_PREFIX, 
                                     "supporting_files/cam2_svm.pkl"))
    
    # Load an existing model
    model1 = joblib.load(os.path.join(PROJECT_PREFIX, "supporting_files/cam1_svm.pkl"))
    print model1
