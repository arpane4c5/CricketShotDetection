#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 00:49:03 2018

@author: Arpan

@Description: Predict the shot boundaries (CUTs) for the entire dataset using a 
pretrained model. Create the JSON file of the predictions.

"""

import os
import numpy as np
import json
from sklearn.externals import joblib
from sklearn import svm

# Local params
DATASET = "/home/hadoop/VisionWorkspace/Cricket/dataset_25_fps"
SUPPORTING_FILES = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files"
HIST_DIFFS = "/home/hadoop/VisionWorkspace/Cricket/scripts/extracted_features/hist_diff_grayscale_ds_25_fps"
META_FILE = "dataset_25_fps_meta_info.json"
MODEL_LIN_SVM = "sbd_model_RF_histDiffs_gray.pkl"      # trained on hist_diffs_grayscale
SBD_TEST_SET_LABELS = "dataset_25_fps_test_set_labels/sbd"

# function to iterate over all the video features of the videos (kept at a loc)
# and predict shot boundary on the features.
# Params: meta_info: dict of meta info for the video dataset.
#       model: path to the trained model (trained on features that are of same type
#       as contained in the featuresFolderPath)
def predict_for_all(featuresFolderPath, meta_info, model_path, stop='all'):
    
    # load the model
    model = joblib.load(model_path)
    vids_keys = meta_info.keys()
    traversed_tot = 0
    cuts = {}
    
    # Iterate over the video keys and get the 
    for i,k in enumerate(vids_keys):
        # read the file of features corresponding to the key k
        feature_file = os.path.join(featuresFolderPath, k.rsplit('.', 1)[0]+'.bin')
        if os.path.isfile(feature_file):
            # read the feature and predict 
            vfeat = np.load(feature_file)       # load feature
            vpreds = model.predict(vfeat)       # make prediction
            idx_preds = np.argwhere(vpreds)     # get indices where 1
            # convert to list of indices where +ve predictions
            cuts[k] = list(idx_preds.reshape(idx_preds.shape[0]))  
            traversed_tot += 1
            print "Done "+str(i+1)+" : "+k
            print str(idx_preds.shape[0])+" / "+str(vfeat.shape[0])
        else:
            print "File not found !! "+k
    
        # to stop after successful traversal of 2 videos, if stop != 'all'
        if stop != 'all' and traversed_tot == stop:
            break

    print "No. of files traversed : "+str(traversed_tot)
    print "No. of files not found : "+str(len(vids_keys)-traversed_tot)
    if traversed_tot == 0:
        print "Check the structure of the dataset folders !!"
    return cuts

    
# calculate the precision, recall and f-measure for the test set sample labels by
# reading the label files one at a time and finding precision, recall, f-measure
# params: test_set_path: contains the labels in json files, one file for each video
#        arranged in a similar fashion as the full dataset.
#        cuts : dict containing the predictions
def  sbd_calculate_accuracy(test_set_path, cuts):
    # calculate metrics
    Nt = 0      # Total no of transitions
    Nc = 0      # No of correctly predicted transitions
    Nd = 0      # No of deletions, not identified as cut
    Ni = 0      # No of insertions, falsely identified as cut
    # Iterate over the json files (of test set samples) kept in subfolders.
    # Calculate the metrics as defined and return the recall, precision and f-measure
    traversed_tot = 0
    sfp_lst = os.listdir(test_set_path)
    for sf in sfp_lst:
        traversed = 0
        sub_src_path = os.path.join(test_set_path, sf)
        if os.path.isdir(sub_src_path):
            # iterate over the json files inside the directory sf
            labelFiles = os.listdir(sub_src_path)
            for labfile in labelFiles:
                if os.path.isfile(os.path.join(sub_src_path, labfile)) and labfile.rsplit('.', 1)[1] in ['json']:
                    with open(os.path.join(sub_src_path, labfile), 'r') as fp:
                        vid_gt = json.load(fp)
                    vid_key = vid_gt.keys()[0]      # only one key in dict is saved
                    gt_list = vid_gt[vid_key]       # list of tuples [[preFNum, postFNum], ...]
                    gt_list = [postFNum for _, postFNum in gt_list]
                    test_list = cuts[vid_key]
                    # Calculate Nt, Nc, Nd, Ni
                    Nt = Nt + len(set(gt_list))
                    Nd = Nd + len(set(gt_list) - set(test_list))
                    Ni = Ni + len(set(test_list) - set(gt_list))
                    Nc = Nc + len(set(gt_list).intersection(set(test_list)))
                    print "Done : "+sf+"/"+labfile
                    print "GT List : "+str(gt_list)
                    print "Predictions : "+ str(test_list)
                    print "Nt = "+str(Nt)
                    print "Nc = "+str(Nc)
                    print "Nd = "+str(Nd)
                    print "Ni = "+str(Ni)            
                    traversed += 1
                    
            traversed_tot += traversed
            
    print "Total files traversed : "+str(traversed_tot)
    # calculate the recall and precision values
    recall = (Nc / (float)(Nc + Nd))
    precision = (Nc / (float)(Nc + Ni))
    f_measure = 2*precision*recall/(precision+recall)
    return [recall, precision, f_measure]


if __name__=="__main__":
    
    # import the dictionary for the meta_information
    with open(os.path.join(SUPPORTING_FILES, META_FILE), 'r') as fp:
        meta_info = json.load(fp)

    cuts_file = "ds25fps_cuts_hist_diffs_gray_rf.json"    
    ###############################################################################
    # Uncomment either this block, if we want to predict from features and save to disk,
    # or load the predictions from disk in the next section...
#    # Predict on all the videos of the dataset
    cuts = predict_for_all(HIST_DIFFS, meta_info, \
                           os.path.join(SUPPORTING_FILES, MODEL_LIN_SVM), stop='all')
#    
    with open(os.path.join(SUPPORTING_FILES, cuts_file), 'w') as fp:
        json.dump(cuts, fp)
    
    ###############################################################################    
    # load the saved cut predictions
    #with open(os.path.join(SUPPORTING_FILES, cuts_file), 'r') as fp:
    #    cuts = json.load(fp)    

    # calculcate accuracy (for CUTs) on the test set sample of the full main dataset.
    print "Predicting on the test set sample set !!"
    test_set_path = os.path.join(SUPPORTING_FILES, SBD_TEST_SET_LABELS)
    [recall, precision, f_measure] = sbd_calculate_accuracy(test_set_path, cuts)
    print "Precision : "+str(precision)
    print "Recall : "+ str(recall)
    print "F-measure : "+str(f_measure)
    
    