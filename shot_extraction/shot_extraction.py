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
import svm_model as svm

# Server Params
# This path contains 4 subfolders : youtube, hotstar_converted, ipl2017, cpl2015
DATASET_PREFIX = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps"  
SUPPORTING_FILES_PATH = "/home/arpan/VisionWorkspace/shot_detection/supporting_files"
CAM1_MODEL = "cam1_svm_model.dat"
CAM2_MODEL = "cam2_svm_model.dat"
HOG_FILE = "hog.xml"
DATASET_INFO = "dataset_25_fps_meta_info.json"
CUTS_INFO = "dataset_25_fps_cut_predictions.json"


class VideoShot:
    '''This class represents a single Video Shot'''
    frameCounter = 0
    framePointer = -1
    shot = None
    
    # constructor
    def __init__(self, srcVideoPath, st_frame, end_frame):
        self.shot = cv2.VideoCapture(srcVideoPath)
        if self.shot.isOpened():
            print True
    
    def addNextFrame(self, frame):
        # add the frame and
        self.frameCounter = self.frameCounter + 1
        return
    
    def displayVideoShot(self):
        return
        
    def save(self, xmlFileName):
        return
    
    # define more functions here for VideoShot class
    

# Iterate over the videos using keys of the meta_info and extract shots for each
# Inputs: meta_info : dictionary for the dataset meta info
# cuts_dict : dictionary for the cut position prediction over entire dataset 
# hog_obj : reference of cv2.HOGDescriptor(xmlFile)
# Output: dictionary with list of tuples representing shots.
def extract_shots_from_all_videos(meta_info, hog_obj, cam1_model, cam2_model, \
                                  cuts_dict, shots_dict):
    # get the list of all videos in the dataset
    srcVideosList = meta_info.keys()
    for idx, vid_key in enumerate(srcVideosList):
        #######################################################################
        # Method 1: Use boundary detection and then starting frame recognition.
        # call a boundary detector
        cuts_list = cuts_dict[vid_key]
        if 1 not in cuts_list:        
            cuts_list.insert(0,1)       # prepend frame 1 to list
        
        vid_shots = get_shots_from_video(DATASET_PREFIX, vid_key, hog_obj, cam1_model, \
                                         cam2_model, cuts_list, meta_info[vid_key])
        if vid_shots is None:
            print "Video_shots list is NULL (None)."
        shots_dict[vid_key] = vid_shots     # save in dictionary
        print "Shots of video "+str(idx)+" : "+vid_key
        print vid_shots

        if idx == 2:    # stop after 3 videos, Comment this to traverse entire ds
            break

        #######################################################################
        # Method 2: Use Shot Proposals, like Action Proposals method
        

# Frame Based Identification (FBI) using HOG descriptor
# Extract the shot from single video
# Inputs: srcVideoFolder and srcVideo define the complete path of video
#       cuts_list --> list of frame nos where boundary is predicted (postFNum)
#       vinfo --> video meta info (dict with partition, dimension and nFrames)
# Output: Returns a list of tuples (st_fr, end_fr) i.e., starting and ending frame
# nos. of the shots.
def get_shots_from_video(srcVideoFolder, srcVideo, hog, cam1_model, cam2_model, \
                         vcuts_list=None, vinfo=None):
    cap = cv2.VideoCapture(os.path.join(srcVideoFolder,srcVideo))
    # if the VideoCapture object is not opened then return None
    if not cap.isOpened():
        print "Error in opening video File !!"
        return None
    # following attributes can be read from vinfo also.
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
                  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nFrames = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
    vid_shots = []
    start_frame = -1
    end_frame = -1
    
    print "Video :: "+srcVideo
    print "Dim :: "+str(dimensions)+"  #### FPS :: "+str(fps)
    
    ncuts = len(vcuts_list)
    ######################
    
    # if cuts_len is empty then ##############
    
    ######################
    # Method 1: Naive method
    
    for i, cut_pos in enumerate(vcuts_list):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cut_pos)
        ret, frame = cap.read()
        
        if ret:
            # convert to grayscale and get HOG feature vector
            hog_vec = hog.compute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cam1_pred = np.int32(cam1_model.predict(hog_vec))
            cam2_pred = np.int32(cam2_model.predict(hog_vec))
            if start_frame == -1:   # if shot has has not begun
                if cam1_pred == 1:  # +ve prediction
                    start_frame = cut_pos
                # else do not test for cam2, since shot has not begun
            elif start_frame >= 0:  # if shot has begun
                if cam2_pred == 0:      # ball is within view, not switched to cam2
                    end_frame = cut_pos-1
                else:
                    if (i+1) < ncuts: # switched to cam2
                        end_frame = vcuts_list[i+1]-1
                    else:
                        end_frame = nFrames-1   # for last cut, ends at end of vid
                vid_shots.append((start_frame, end_frame))  
                start_frame = end_frame = -1
            
            
################################################################################
################################################################################                    
        
        #writeShotToFile(srcVideo, start_index, stop_index, outFileName)
#    ret, prev_frame = cap.read()
#    #prev_frame = prev_frame[50:310,100:540]   # Cropping
#    # convert frame to GRAYSCALE
#    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#    
#    while(cap.isOpened()):
#        ret, frame = cap.read()
#
#        if ret:
#            
#            #fgmask = fgbg.apply(frame)
#            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
################################################################################
#            # decide whether frame is part of shot or not
#            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
#            
#            # Append value to a list and come up with a decision algo 
#            # for categorizing shot into a cricket shot
#
################################################################################            

    cap.release()
    
    return vid_shots


# Decide whether a video frame is a part of the positive example video shot
# Take features from the frame and predict using a trained model
def predict_frame_class(frame, cam=1):
    #import svm_model
    # Extract features of the frame (HOG etc) and
    # Predict using the trained SVM model 
    r = np.random.randint(2)    # coin toss, bernoulli RV
    if r==1:
        return True
    else:
        return False


###############################################################################

# method to predict a sequence of frames and check for a sequence of True values
# eg. T T T T F F T F F F F F ....
def method2():
    return 

###############################################################################
###############################################################################



def display_mat_stats(mat):
    print "Shape : "+str(mat.shape)+"  ::  Type : "+str(type(mat))
    print "Sum of values : "+str(np.sum(mat))
    print "Mean : "+str(np.mean(mat))+"  ::  SD : "+str(np.std(mat))  
    #print mat
    return

def waitTillEscPressed():
    while(True):
        if cv2.waitKey(10)==27:
            print("Esc Pressed")
            return
            

if __name__=='__main__':
    # path for sample dataset on localhost
    #srcVideoFolder = "/home/hadoop/VisionWorkspace/VideoData/ICC WT20"
    
    # read the meta info from meta_info file
    with open(os.path.join(SUPPORTING_FILES_PATH, DATASET_INFO), "r") as fp:
        meta_info = json.load(fp)
    
    # create the cv2.HOGDescriptor object to be applied to grayscale images
    hog_obj = cv2.HOGDescriptor(os.path.join(SUPPORTING_FILES_PATH, HOG_FILE))
    # load the pretrained cam1 and cam2 svm models
    cam1_model = svm.SVM()
    cam1_model.load(os.path.join(SUPPORTING_FILES_PATH, CAM1_MODEL))
    cam2_model = svm.SVM()
    cam2_model.load(os.path.join(SUPPORTING_FILES_PATH, CAM2_MODEL))
    
    # read the cut predictions json file.
    with open(os.path.join(SUPPORTING_FILES_PATH, CUTS_INFO), 'r') as fp:
        cuts_dict = json.load(fp)
    
    
    shots_dict = {}
    
    # iterate over all videos to extract HOG features 
    # OR extract and mark one video at a time
    extract_shots_from_all_videos(meta_info, hog_obj, cam1_model, cam2_model, \
                                  cuts_dict, shots_dict)    
    #extract_shot_from_video(srcVideoFolder, srcVideosList[0])
    
    # write shots_dict to disk
    
    
###############################################################################    
    #zoom_detect(os.path.join(srcVideoFolder, srcVideosList[0]))
    #check_sift("/home/hadoop/VisionWorkspace/ActivityProjPy/frames_col/f99.jpg", \
    #       "/home/hadoop/VisionWorkspace/ActivityProjPy/frames_col/f112.jpg")