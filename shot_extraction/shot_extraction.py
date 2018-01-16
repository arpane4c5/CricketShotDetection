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
DATASET_PREFIX = "/home/arpan/DATA_Drive/Cricket/dataset/"
SUPPORTING_FILES_PATH = "/home/arpan/VisionWorkspace/shot_detection/supporting_files"
CAM1_MODEL = "cam1_svm.dat"
CAM2_MODEL = "cam2_svm.dat"
DATASET_INFO = "dataset_partitions_info.json"


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
    

# Iterate over the videos kept in a folder and extract relevant shots from each
# Input: dictionary to the dataset meta info
def extract_shots_from_all_videos(meta_info):
    # get the list of videos
    srcVideosList = os.listdir(srcVideoFolder)
    for video in srcVideosList:
        # call a boundary detector
        cuts_list = get_cuts_list_for_video(video)
        if 1 not in cuts_list:        
            cuts_list.insert(0,1)       # prepend frame 1 to list
        
        extract_shot_from_video(srcVideoFolder, video, cuts_list)


# Frame Based Identification (FBI) using HOG descriptor
# Extract the shot from single video
# Inputs: srcVideoFolder and srcVideo define the complete path of video
#       cuts_list --> list of frame nos where boundary is predicted (postFNum)
def extract_shot_from_video(srcVideoFolder, srcVideo, cuts_list):
    cap = cv2.VideoCapture(os.path.join(srcVideoFolder,srcVideo))
    # if the VideoCapture object is not opened then exit
    if not cap.isOpened():
        import sys
        print "Error in opening video File !!"
        sys.exit(0)

    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_no_of_frames = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
    start_frame = -1
    end_frame = -1
    cam1_flag = False   # True is for positive class of frame
    cam2_flag = False   # True if first frame of camera 2 is detected when
                        # shot is played, checked only after camera1 flag is true
    
    print "Video :: "+srcVideo
    print "Dim :: "+str(dimensions)+"  #### FPS :: "+str(fps)
    # see if it matches 360x640 and ~25 FPS
    no_of_cuts = len(cuts_list)
    
    for i in range(no_of_cuts):
        cut_pos = cuts_list[i]
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, cut_pos)
        ret, frame = cap.read()
        
        if ret:
            if not cam1_flag:
                if cam2_flag and not start_frame==-1:
                    end_frame = cut_pos
                    # call write_shot
                    start_frame = end_frame = -1
                    cam2_flag = False
                    
                cam1_flag = predict_frame_class_cam1(frame)
                if cam1_flag:
                    start_frame = cut_pos
                    continue
            else:
                cam1_flag = False
                cam2_flag = predict_frame_class_cam2(frame)
                if cam2_flag:
                    # check for last cut value
                    continue
                else:
                    end_frame = cut_pos
                    # call write_shot
                    start_frame = end_frame = -1
                    
                    
                
#    
        
        #writeShotToFile(srcVideo, start_index, stop_index, outFileName)
#    ret, prev_frame = cap.read()
#    #prev_frame = prev_frame[50:310,100:540]   # Cropping
#    # convert frame to GRAYSCALE
#    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#    
#    # If frame is not of a const dimension 360x640 then scale
#    # frame = cv2.resize(frame, (640, 360), intepolation = cv2.INTER_CUBIC)
#    # check FPS also, it should be ~25
#    while(cap.isOpened()):
#        ret, frame = cap.read()
#
#        if ret:
#            # Do the necessary processing here
#            #frame = imutils.resize(frame, width=600)
#            
#            #fgmask = fgbg.apply(frame)
#            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
################################################################################
#            # decide whether frame is part of shot or not
#            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
#
################################################################################ 
#            
################################################################################ 
#             
#            frame_res = is_positive_shot_frame(curr_frame)
#            
#            # Append value to a list and come up with a decision algo 
#            # for categorizing shot into a cricket shot
#            
#            cv2.imshow("Video", frame)            
#
################################################################################            
#            
#            #cv2.imshow('Cropped', frame[50:310,100:540])
#            #cv2.imshow("background", mask)
#            waitTillEscPressed()
#            #if cv2.waitKey(10)==27:
#            #    print("Esc Pressed")
#            #    break
#            frameCounter = frameCounter + 1
#            prev_frame = curr_frame
#        else:
#            break
    cap.release()
    cv2.destroyAllWindows()
    
    return 


# Decide whether a video frame is a part of the positive example video shot
# Take features from the frame and predict using a trained model
def is_positive_shot_frame(frame):
    #import svm_model

    
    # Extract features of the frame (HOG etc) and
    # Predict using the trained SVM model 
    r = np.random.randint(2)    # coin toss, bernoulli RV
    if r==1:
        return True
    else:
        return False


# function in shot_detection.py 
# def write_shot_to_file(videoShotObject, destPath):
    # write the file to the 
    # No need to hold all the frames of shot in memory (as part of object)
    # shot object can only hold the meta information
#    return    

###############################################################################
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
    
    # iterate over all videos to extract HOG features 
    # OR extract and mark one video at a time
    
    extract_shots_from_all_videos(meta_info)    
    #extract_shot_from_video(srcVideoFolder, srcVideosList[0])
    
    # Iterate over all the videos of dataset and 
    # Need extracted features (HOG) for prediction on 
###############################################################################    
    #zoom_detect(os.path.join(srcVideoFolder, srcVideosList[0]))
    #check_sift("/home/hadoop/VisionWorkspace/ActivityProjPy/frames_col/f99.jpg", "/home/hadoop/VisionWorkspace/ActivityProjPy/frames_col/f112.jpg")