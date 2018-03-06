#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:43:26 2017

@author: Arpan

@Description: Detect shot boundaries in the Highlight videos dataset (WC T20 2016)
using weighted chi squared features.

"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import SBD_xml_operations as lab_xml
from sklearn import svm
import pickle

# Server params
#DATASET = "/opt/datasets/cricket/ICC_WT20"
#LABELS = "/opt/datasets/cricket/gt_ICC_WT20"

# Local params
DATASET = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
LABELS = "/home/hadoop/VisionWorkspace/ActivityProjPy/gt_ICC_WT20"

# Split the dataset files into training, validation and test sets
def split_dataset_files():
    filenames = sorted(os.listdir(DATASET))         # read the filenames
    # filenames = [t.rsplit('.')[0] for t in filenames]   # remove the extension
    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]
    
    
# function to extract the features from a list of videos
# Params: vids_lst = list of videos for which hist_diff values are to be extracted
# Return: hist_diff_all = f values of histogram diff each (256 X C) (f is no of frames)
def extract_hist_diff_vids(vids_lst):
    # iterate over the videos to extract the hist_diff values
    hist_diff_all = []
    for idx, vid in enumerate(vids_lst):
        #get_hist_diff(os.path.join(DATASET, vid+'.avi'))
        diffs = getWtChiSqHistogramOfVideo(os.path.join(DATASET, vid+'.avi'))
        #print "diffs : ",diffs
        print "Done : " + str(idx+1)
        hist_diff_all.append(diffs)
        # save diff_hist to disk    
        #outfile = file(os.path.join(destPath,"diff_hist.bin"), "wb")
        #np.save(outfile, diffs)
        #outfile.close()    
        #break
    return hist_diff_all

# Function taken from extract_sqchi_diffs.py file.
# function to get the Weighted Chi Squared distances of histograms 
# for getting the grayscale histogram differences, uncomment two lines
# Copied and editted from SBD_detection_methods.py script
# color=('b') : For grayscale, ('b','g','r') for RGB
def getWtChiSqHistogramOfVideo(srcVideoPath, color=('b','g','r')):
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    
    N = 256     # no of bins for histogram
    frameCount = 0
    # uncomment following line for grayscale
    #color = ('b')
    prev_hist = np.zeros((256, len(color)))      # histograms for 3 channels
    curr_hist = np.zeros((256, len(color)))
    diffs = np.zeros((1, len(color)))        # single number appended to vector
    while(cap.isOpened()):
        # Capture frame by frame
        ret, frame = cap.read()
        # print(ret)
    
        if ret:
            # frame = cv2.flip(frame)
            frameCount = frameCount + 1
            
            # condition for converting frames to grayscale
            # color = ('b')
            if len(color) == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         
            
            for i,col in enumerate(color):
                # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
                curr_hist[:,i] = np.reshape(cv2.calcHist([frame], [i], None, [N], [0,N]), (N,))
            
            if frameCount > 1:
                # find the L1 distance of the current frame hist to previous frame hist 
                #dist = np.sum(abs(curr_hist - prev_hist), axis=0)
                ######################################################
                # find the squared distances
                sq_diffs = (curr_hist - prev_hist)**2
                # max values of corresponding bins max(Hi(k), Hj(k))
                max_bin_values = np.maximum(curr_hist, prev_hist)
                # find sq_diffs/max_bin_values, will generate nan for 0/0 cases
                d1 = np.divide(sq_diffs, max_bin_values)
                # convert nan to 0
                nan_indices = np.isnan(d1)
                d1[nan_indices] = 0
                # Identify which channel is RGB
                # multiply col 0 by gamma (Blue), col 1 by beta (green)
                # and col 2 by alpha (red)
                if len(color)==3:       # Weighing the BGR channels
                    gamma_beta_alpha = np.array([0.114, 0.587, 0.299])
                else:           # Not weighing the grayscale/other channels
                    gamma_beta_alpha = np.ones(len(color))
                # multiply each column by const
                d1 = d1*gamma_beta_alpha
                #print np.sum(d1)
                #d1 = np.sum(np.sum(d1, axis=1))     # sum the columns (3 channels)
                #sum up everything
                dist = np.sum(d1, axis=0)/(len(color)*N)       # N = 256 bins
                #print "Dist shape : "+str(dist.shape)
                ######################################################
                #diffs.append(dist)
                diffs = np.vstack([diffs, dist])
                #print("dist appended = ", type(dist), dist)
                #print("diffs shape = ", type(diffs), diffs.shape)
        
            np.copyto(prev_hist, curr_hist)        
            
            if cv2.waitKey(10) == 27:
                print('Esc pressed')
                break        
        else:
            break

    # When everything done, release the capture
    cap.release()
    return diffs


# Visualize the positive and negative samples
# Params: list of numpy arrays of size nFrames-1 x Channels
def visualize_feature(samples_lst, title="Histogram", bins=300):
    
    if len(samples_lst) == 1:
        print "Cannot Visualize !! Only single numpy array in list !!"
        return
    elif len(samples_lst) > 1:
        sample = np.vstack((samples_lst[0], samples_lst[1]))
    
    # Iterate over the list to vstack those and get a single matrix
    for idx in range(2, len(samples_lst)):
        sample = np.vstack((sample, samples_lst[idx]))
        
    vals = list(sample.reshape(sample.shape[0]))
    
    plt.hist(vals, normed=True, bins=bins)
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("Frequency")

        
def create_dataframe(pos_samples, neg_samples):
    # create a r X 2 matrix with 2nd column of 1s (for pos_samples) or 0s (neg_samples)
    pos_feats = pos_samples[0]
    for i in range(1, len(pos_samples)):
        pos_feats = np.vstack((pos_feats, pos_samples[i]))
    pos_feats = np.hstack((pos_feats, np.ones((pos_feats.shape[0], 1))))
    
    # create similarly for negative samples
    neg_feats = neg_samples[0]
    for i in range(1, len(neg_samples)):
        neg_feats = np.vstack((neg_feats, neg_samples[i]))
    neg_feats = np.hstack((neg_feats, np.zeros((neg_feats.shape[0], 1))))
    
    # change this depending on the no. of features 
    df = pd.DataFrame(np.vstack((pos_feats, neg_feats)), columns=["X1", "X2", "X3", "Y"])
    print "Dataframe Unshuffled : "
    print df.head()
    # shuffle dataframe
    df = df.sample(frac=1)  # .reset_index(drop=True)
    print df.head()
    return df
    
# given a dataframe in the form [X1, X2, .., Y] with Y being binary, train a model
def train_model1(df_train):
    clf = svm.SVC(kernel = 'linear')
    # 
    clf.fit(df_train.loc[:, df_train.columns != 'Y'], df_train.loc[:,'Y'])
    return clf
    
# calculate the precision, recall and f-measure for the validation of test set
# params: preds_dict: {"vid_name": [98, 138, ...], ...}
def  calculate_accuracy(preds_dict, split = "val"):
    # calculate metrics
    Nt = 0      # Total no of transitions
    Nc = 0      # No of correctly predicted transitions
    Nd = 0      # No of deletions, not identified as cut
    Ni = 0      # No of insertions, falsely identified as cut
    # Iterate over the xml files (keys of preds_dict) and corresponding gt xml
    # Calculate the metrics as defined and return the recall, precision and f-measure
    for i,fname in enumerate(preds_dict.keys()):
        gt_list = lab_xml.get_cuts_list_from_xml(os.path.join(LABELS, 'gt_'+fname+'.xml'))
        test_list = preds_dict[fname]

        # Calculate Nt, Nc, Nd, Ni
        Nt = Nt + len(set(gt_list))
        Nd = Nd + len(set(gt_list) - set(test_list))
        Ni = Ni + len(set(test_list) - set(gt_list))
        Nc = Nc + len(set(gt_list).intersection(set(test_list)))
        
        print gt_list
        print test_list        
        print "Nt = "+str(Nt)
        print "Nc = "+str(Nc)
        print "Nd = "+str(Nd)
        print "Ni = "+str(Ni)
        
    # calculate the recall and precision values
    recall = (Nc / (float)(Nc + Nd))
    precision = (Nc / (float)(Nc + Ni))
    f_measure = 2*precision*recall/(precision+recall)
    return [recall, precision, f_measure]

# function to predict the cuts on the validation or test videos
def make_predictions(vids_lst, model, split = "val"):
    # extract the hist diff features and return as a list entry for each video in vids_lst
    hist_diffs = extract_hist_diff_vids(vids_lst)
    # form a dictionary of video names (as keys) and corresponding list of hist_diff values
    #hist_diffs_dict = dict(zip(vids_lst, hist_diffs))
    print "Extracted features !! "
    
    preds = {}
    # make predictions using the model
    for idx, vname in enumerate(vids_lst):
        # make predictions using the model (returns a 1D array of 0s and 1s)
        vpreds = model.predict(hist_diffs[idx])
        # gives a n x 1 array of indices where n non-zero val occurs
        idx_preds = np.argwhere(vpreds)
        idx_preds = list(idx_preds.reshape(idx_preds.shape[0]))
        preds[vname] = idx_preds    # list of indices for positive predictions
        print(vname, idx_preds)
    
    return calculate_accuracy(preds)

if __name__=="__main__":
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = split_dataset_files()
    #print(train_lst, len(train_lst))
    #print 100*"-"
    #print val_lst
    #print 100*"-"
    #print test_lst
    
#    # Extract the histogram difference features from the training set
#    hist_diffs_train = extract_hist_diff_vids(train_lst)
#    
#    train_lab = ["gt_"+f+".xml" for f in train_lst]
#    val_lab = ["gt_"+f+".xml" for f in val_lst]
#    test_lab = ["gt_"+f+".xml" for f in test_lst]
#    
#    # get the positions where cuts exist for training set
#    cuts_lst = []
#    for t in train_lab:
#        cuts_lst.append(lab_xml.get_cuts_list_from_xml(os.path.join(LABELS, t)))
#        #print cuts_lst
#    
#    pos_samples = []
#    neg_samples = []
#    
#    for idx, sample in enumerate(hist_diffs_train):
#        pos_samples.append(sample[cuts_lst[idx],:])     # append a np array of pos samples
#        neg_indices = list(set(range(len(sample))) - set(cuts_lst[idx]))
#        # subset
#        neg_samples.append(sample[neg_indices,:])
#    
#    
#    # Save the pos_samples and neg_samples lists to disk
#
#    with open("pos_samples_chiSq.pkl", "wb") as fp:
#        pickle.dump(pos_samples, fp)
#        
#    with open("neg_samples_chiSq.pkl", "wb") as fp:
#        pickle.dump(neg_samples, fp)
        
    # Read the lists from disk to the pickle files
    with open("pos_samples_chiSq.pkl", "rb") as fp:
        pos_samples = pickle.load(fp)
    
    with open("neg_samples_chiSq.pkl", "rb") as fp:
        neg_samples = pickle.load(fp)
        
    #print "Visualizing positive and negative training samples ..."
    #visualize_feature(pos_samples, "Positive Samples", 30)
    #visualize_feature(neg_samples, "Negative Samples", 300)
    
    df = create_dataframe(pos_samples, neg_samples)
    print df.shape
    
    # Training a model given a dataframe
    print "Training SVM model ..."
    svm_model = train_model1(df)
    
    # get predictions on the validation or test set videos
    #pr = svm_model.predict(df.sample(frac=0.001).loc[:,['X']])
    
    # extract the validation/test set features and make predictions on the same
    print "Predicting on the validation set !!"
    [recall, precision, f_measure] = make_predictions(val_lst, svm_model)
    print "Precision : "+str(precision)
    print "Recall : "+ str(recall)
    print "F-measure : "+str(f_measure)
    
    
    #######################################################
    # Extend 1:
    # Add feature: #Frames till the last shot boundary: Will it be correct feature
    # How to handle testing set feature. A single false +ve will screw up the subsequent 
    # predictions.
    
    #######################################################
    
    # Extend 2: Experiment with Random Forests, decision trees and Bayesian Inf
    #
    
    #######################################################
    
    # Extend 3 : Learn a CNN architecture
    
    #######################################################
    
    # Extend 4 : Learn an RNN by extracting features 