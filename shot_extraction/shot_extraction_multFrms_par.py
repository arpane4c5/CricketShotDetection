#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 00:21:18 2017

@author: Arpan

Description: Shot extraction
"""

import cv2
import numpy as np
import os
import json
import pandas as pd
import time
from sklearn import svm
from joblib import Parallel, delayed
from sklearn.externals import joblib as jl

# Server Params
# This path contains 4 subfolders : youtube, hotstar_converted, ipl2017, cpl2015
#DATASET_PREFIX = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps"  
#SUPPORTING_FILES_PATH = "/home/arpan/VisionWorkspace/shot_detection/supporting_files"
# Local Params
DATASET_PREFIX = "/home/hadoop/VisionWorkspace/Cricket/dataset_25_fps"  
SUPPORTING_FILES_PATH = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files"
CAM1_MODEL = "cam1_svm.pkl"
CAM2_MODEL = "cam2_svm.pkl"
HOG_FILE = "hog.xml"
DATASET_INFO = "dataset_25_fps_meta_info.json"
CUTS_INFO = "ds25fps_cuts_hist_diffs_gray_rf.json"

# Iterate over the videos using keys of the meta_info and extract shots for each
# Inputs: meta_info : dictionary for the dataset meta info
# cuts_dict : dictionary for the cut position prediction over entire dataset 
# hog_obj : reference of cv2.HOGDescriptor(xmlFile)
# Output: dictionary with list of tuples representing shots.
def extract_shots_from_all_videos(meta_info, cuts_dict, shots_dict):
    # get the list of all videos in the dataset
    srcVideosList = meta_info.keys()
    # prepend pos of 1st frame if it doesn't exist in the cuts_list
    cuts_all = {}
    for k,v in cuts_dict.iteritems():
        if 1 not in v:
            v.insert(0, 1)  # prepend frame 1 to list
        cuts_all[k] = v
    nCuts = [len(cuts_all[vid_key]) for vid_key in srcVideosList]
    nCuts_df = pd.DataFrame({"keys":srcVideosList, "nCuts":nCuts})
    nCuts_df = nCuts_df.sort_values(["nCuts"], ascending=True)  # sort by nCuts
    nCuts_df = nCuts_df.reset_index(drop=True)  # reset row labels
    
    #######################################################################
    # Method 1: Use boundary detection and then starting frame recognition.
    # call a boundary detector

    #### Form the pandas Dataframe and parallelize over the files.
    nrows = nCuts_df.shape[0]
    batch = 50      # No. of videos in a batch.
    njobs = 10      # No. of threads.
    
    for i in range(nrows/batch):
        # vid_key is nCuts_df['keys'][i*batch+j]
        batch_segments = Parallel(n_jobs=njobs)(delayed(get_shots_from_video) \
                          (DATASET_PREFIX, nCuts_df['keys'][i*batch+j], \
                           cuts_all[nCuts_df['keys'][i*batch+j]], \
                            meta_info[nCuts_df['keys'][i*batch+j]]) \
                          for j in range(batch))
        
        # Writing the diffs in a serial manner
        for j in range(batch):
            if batch_segments[j] is not None:
                shots_dict[nCuts_df['keys'][i*batch+j]] = batch_segments[j]
                print "Written "+str(i*batch+j+1)+" : "+ \
                                    nCuts_df['keys'][i*batch+j]
            
    # For last batch which may not be complete, extract serially
    last_batch_size = nrows - ((nrows/batch)*batch)
    if last_batch_size > 0:
        batch_segments = Parallel(n_jobs=njobs)(delayed(get_shots_from_video) \
                              (DATASET_PREFIX, nCuts_df['keys'][(nrows/batch)*batch+j], \
                               cuts_all[nCuts_df['keys'][(nrows/batch)*batch+j]], \
                                meta_info[nCuts_df['keys'][(nrows/batch)*batch+j]]) \
                              for j in range(last_batch_size)) 
        # Writing the diffs in a serial manner
        for j in range(last_batch_size):
            if batch_segments[j] is not None:
                shots_dict[nCuts_df['keys'][(nrows/batch)*batch+j]] = batch_segments[j]
                print "Written "+str((nrows/batch)*batch+j+1)+" : "+ \
                                    nCuts_df['keys'][(nrows/batch)*batch+j]

        #######################################################################
        # Method 2: Use Shot Proposals, like Action Proposals method
        

# Frame Based Identification (FBI) using HOG descriptor
# Extract the shot from single video
# Inputs: srcVideoFolder and srcVideo define the complete path of video
#       cuts_list --> list of frame nos where boundary is predicted (postFNum)
#       vinfo --> video meta info (dict with partition, dimension and nFrames)
# Output: Returns a list of tuples (st_fr, end_fr) i.e., starting and ending frame
# nos. of the shots.
def get_shots_from_video(srcVideoFolder, srcVideo, vcuts_list=None, vinfo=None):
    global hog, cam1_model, cam2_model
    cap = cv2.VideoCapture(os.path.join(srcVideoFolder,srcVideo))
    # if the VideoCapture object is not opened then return None
    if not cap.isOpened():
        print "Error in opening video File !!"
        return None
    # following attributes can be read from vinfo also.
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
                  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nFrames = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    vid_shots = []
    start_frame = -1
    end_frame = -1
    
    ncuts = len(vcuts_list)
    ######################
    # if cuts_len is empty then ##############
    ######################
    # Method 1: Naive method
    
    for i, cut_pos in enumerate(vcuts_list):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cut_pos)
        #print "Pos : "+str(cut_pos)
        ret, frame = cap.read()
        #cv2.imshow("prev", frame)
        #ret, frame = cap.read()
        #cv2.imshow("Current", frame)
        #cv2.waitKey(0)
        
        if ret:
            # convert to grayscale and get HOG feature col vector (79380,1)
            hog_vec = hog.compute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            # reshape into a row vector to pass to predict()
            cam1_pred = np.int32(cam1_model.predict(hog_vec.reshape((-1, hog_vec.shape[0]))))
            cam2_pred = np.int32(cam2_model.predict(hog_vec.reshape((-1, hog_vec.shape[0]))))
            if start_frame == -1:   # if shot has has not begun
                if cam1_pred[0] == 1:  # +ve prediction
                    start_frame = cut_pos
                # else do not test for cam2, since shot has not begun
            elif start_frame >= 0:  # if shot has begun
                if cam2_pred[0] == 0:      # ball is within view, not switched to cam2
                    end_frame = cut_pos-1
                else:
                    if (i+1) < ncuts: # switched to cam2
                        end_frame = vcuts_list[i+1]-1
                    else:
                        end_frame = nFrames-1   # for last cut, ends at end of vid
                vid_shots.append((start_frame, end_frame))  
                start_frame = end_frame = -1
            
###############################################################################

    cap.release()
    return vid_shots


###############################################################################

# method to predict a sequence of frames and check for a sequence of True values
# eg. T T T T F F T F F F F F ....
def method2():
    return 

###############################################################################
###############################################################################

if __name__=='__main__':
    global hog, cam1_model, cam2_model
    # read the meta info from meta_info file
    with open(os.path.join(SUPPORTING_FILES_PATH, DATASET_INFO), "r") as fp:
        meta_info = json.load(fp)
    
    # create the cv2.HOGDescriptor object to be applied to grayscale images
    hog = cv2.HOGDescriptor(os.path.join(SUPPORTING_FILES_PATH, HOG_FILE))
    
    # load the pretrained cam1 and cam2 svm models
    cam1_model = jl.load(os.path.join(SUPPORTING_FILES_PATH, CAM1_MODEL))
    cam2_model = jl.load(os.path.join(SUPPORTING_FILES_PATH, CAM2_MODEL))
    
    # read the cut predictions json file.
    with open(os.path.join(SUPPORTING_FILES_PATH, CUTS_INFO), 'r') as fp:
        cuts_dict = json.load(fp)
    
    shots_dict = {}
    start = time.time()
    # iterate over all videos to extract HOG features 
    # OR extract and mark one video at a time
    extract_shots_from_all_videos(meta_info, cuts_dict, shots_dict)    
    end = time.time()
    print "Total execution time : "+str(end-start)
    
    # write shots_dict to disk
    shots_filename = "cricShots_hdiffGray_multFrms.json"
    with open(os.path.join(SUPPORTING_FILES_PATH, shots_filename), 'w') as fp:
        json.dump(shots_dict, fp)
    
###############################################################################    
