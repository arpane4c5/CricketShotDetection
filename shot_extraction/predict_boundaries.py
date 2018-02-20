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
MODEL_LIN_SVM = "sbd_LinSVM_histDiffs.pkl"      # trained on hist_diffs_grayscale

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

    
## calculate the precision, recall and f-measure for the validation of test set
## params: preds_dict: {"vid_name": [98, 138, ...], ...}
#def  calculate_accuracy(preds_dict, split = "val"):
#    # calculate metrics
#    Nt = 0      # Total no of transitions
#    Nc = 0      # No of correctly predicted transitions
#    Nd = 0      # No of deletions, not identified as cut
#    Ni = 0      # No of insertions, falsely identified as cut
#    # Iterate over the xml files (keys of preds_dict) and corresponding gt xml
#    # Calculate the metrics as defined and return the recall, precision and f-measure
#    for i,fname in enumerate(preds_dict.keys()):
#        gt_list = lab_xml.get_cuts_list_from_xml(os.path.join(LABELS, 'gt_'+fname+'.xml'))
#        test_list = preds_dict[fname]
#
#        # Calculate Nt, Nc, Nd, Ni
#        Nt = Nt + len(set(gt_list))
#        Nd = Nd + len(set(gt_list) - set(test_list))
#        Ni = Ni + len(set(test_list) - set(gt_list))
#        Nc = Nc + len(set(gt_list).intersection(set(test_list)))
#        
#        print gt_list
#        print test_list        
#        print "Nt = "+str(Nt)
#        print "Nc = "+str(Nc)
#        print "Nd = "+str(Nd)
#        print "Ni = "+str(Ni)
#        
#    # calculate the recall and precision values
#    recall = (Nc / (float)(Nc + Nd))
#    precision = (Nc / (float)(Nc + Ni))
#    f_measure = 2*precision*recall/(precision+recall)
#    return [recall, precision, f_measure]


if __name__=="__main__":
    
    # improt the dictionary for the meta_information
    with open(os.path.join(SUPPORTING_FILES, META_FILE), 'r') as fp:
        meta_info = json.load(fp)
    
    # Predict on all the videos of the dataset
    cuts = predict_for_all(HIST_DIFFS, meta_info, \
                           os.path.join(SUPPORTING_FILES, MODEL_LIN_SVM), stop='all')
    
    cuts_file = "ds25fps_cuts_hist_diffs_gray_lsvm.json"
    with open(os.path.join(SUPPORTING_FILES, cuts_file), 'w') as fp:
        json.dump(cuts, fp)
    
    # extract the validation/test set features and make predictions on the same
    #print "Predicting on the validation set !!"
    #[recall, precision, f_measure] = make_predictions(val_lst, svm_model)
    #print "Precision : "+str(precision)
    #print "Recall : "+ str(recall)
    #print "F-measure : "+str(f_measure)
    
    
    