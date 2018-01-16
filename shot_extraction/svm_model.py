# -*- coding: utf-8 -*-
"""
Created on Sun May 21 01:59:42 2017

@author: Arpan
Description: Script to train simple ML models on camera frames. We use HOG features 
extracted from positive and negative frames. The models are saved on disk and used 
to extract cricket shots by detecting the starting frames as positive examples.
"""

import cv2
import numpy as np

CAM1_FRAMES_DATA = "/home/hadoop/VisionWorkspace/ActivityProjPy/ExtractFrames/cam1_frames"
CAM2_FRAMES_DATA = "/home/hadoop/VisionWorkspace/ActivityProjPy/ExtractFrames/cam2_frames"
CLASS_N = 2
# refer the opencv sample program on SVM training for details of params
svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, \
            C=2.67, gamma = 5.383)

class StatModel(object):
    def load(self, fn):
        self.model.load(fn) #Known bug: github.com/opencv/opencv/issues/4969     
    def save(self, fn):
        self.model.save(fn)

# SVM class
# https://stackoverflow.com/questions/8687885/python-opencv-svm-implementation
class SVM(StatModel):
    '''Wrapper for Support Vector Machine'''
    def __init__(self):
        self.model = cv2.SVM()
        
    def train(self, samples, responses, params):
        # setting algo params
        self.model.train(samples, responses, params=params)
        
    def predict(self, samples):
        return  np.float32([self.model.predict(s) for s in samples] )


# read folder images paths and interpret labels from their names
# function returns the list of image paths and corresponding labels
def load_data(srcDataPath):
    img_list = list()
    labels_list = list()
    # read the images from the folder 
    import os
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
    hog = cv2.HOGDescriptor("hog.xml")
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
    resp = model.predict(samples)
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
        
    
    # Train a model    
    model = SVM()   # C = 2.67, gamma = 5.383
    model.train(samples_train, labels_train, svm_params)    
    
    # Evaluate the algorithm
    print "Evaluation on the test data:"
    evaluate_model(model, samples_test, labels_test)
    
    return model
    

if __name__=='__main__':
    
    # Train the model on 
    model1 = train_model(CAM1_FRAMES_DATA, 321)
    # save model to disk
    model1.save("cam1_svm_model.dat")
    
    # Save the model to disk
    model2 = train_model(CAM2_FRAMES_DATA, 456)
    model2.save("cam2_svm_model.dat")
    
    # Load an existing model
    # model = SVM
    # model.load("cam1_svm_model.dat")
