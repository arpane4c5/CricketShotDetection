#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:24:28 2017

@author: Arpan

Description: Create json meta info files for the dataset. Decide the training, validation 
and testing partition proportions and videos.

"""

import cv2
import os
import json
import random

# Server params
DATASET = "/home/arpan/DATA_Drive/Cricket/dataset"
LABELS = ""

notOpened, opened = 0, 0
# Local params
#DATASET = "/home/hadoop/VisionWorkspace/VideoData/ICC WT20"
#LABELS = "/home/hadoop/VisionWorkspace/ActivityProjPy/gt_ICC_WT20"


# function to read the files and create the json meta info files.
def create_meta_files():
    d = {}
    # read the files and calculate the meta information
    subfolders = os.listdir(DATASET)
    for sf in subfolders:
        for fl in os.listdir(os.path.join(DATASET, sf)):    # iterate over files
            #write_meta_data(os.path.join(DATASET,sf, fl))
            d[sf+"/"+fl] = read_meta_info(os.path.join(DATASET, sf, fl))
            
    print "Files Opened: {}, Not Opened: {}".format(opened, notOpened)
    return d


def read_meta_info(srcVid):
    global notOpened, opened
    cap = cv2.VideoCapture(srcVid)
    if not cap.isOpened():
        import sys
        notOpened +=1
        print "Not Opened !! Abort !! "+srcVid
        sys.exit(0)
        
    dimensions = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    opened +=1 
    cap.release()
    print "{} File: {}".format(opened, srcVid)
    print "\t Dim: {}, FPS: {}, #Frames: {}".format(dimensions, fps, nFrames)
    
    return {"dimensions":dimensions, "fps": fps, "nFrames": nFrames}

# function to take the video information dictionary and get total time and 
# total no of frames present in the dataset.
def get_total_time(meta_info_dict):
    totalTime = 0
    totalFrames = 0
    
    for k,v in meta_info_dict.iteritems():
        totalTime += v["nFrames"]/v["fps"]
        totalFrames += v["nFrames"]
    
    print "Total Frames = "+str(totalFrames)
    print "Total Time = "+str(totalTime)
    
    return totalFrames, totalTime

# function to take the video information and calculate the validation and test sets
def calculate_partitions(meta_info_dict, tot_time):
    
    keys = meta_info_dict.keys()
    random.shuffle(keys)    # works in-place and returns None
    # taking only 20% of the total time of videos (not on 20% of #videos)
    target_test_time = 0.2*tot_time
    test_time = 0
    # assign the "partition" key as "testing" for specific randomly chosen video
    while test_time < target_test_time:
        k = keys.pop()
        meta_info_dict[k]["partition"] = "testing"
        test_time += (meta_info_dict[k]["nFrames"]/meta_info_dict[k]["fps"])
    
    # For validation videos
    val_time = 0
    while val_time < target_test_time:
        k = keys.pop()
        meta_info_dict[k]["partition"] = "validation"
        val_time += (meta_info_dict[k]["nFrames"]/meta_info_dict[k]["fps"])
        
    # Assign remaining to training set
    for k in keys:
        meta_info_dict[k]["partition"] = "training"
    
    # write the file to disk
    with open("dataset_partitions_info.json", "w") as fp:
        json.dump(meta_info_dict, fp)
    
    print "Total Time of all videos : {} \n \
    Time for validation set videos (20%) : {} \n \
    Time for test set videos (20%) : {} ".format(tot_time, val_time, test_time)
    print "Partitioning done and saved to json file !!"


if __name__=="__main__":
    
    #meta_info = create_meta_files()
    
    #with open("dataset_meta_info.json", "w") as fp:
    #    json.dump(meta_info, fp)
    
    with open("dataset_meta_info.json", "r") as fp:
        meta_info = json.load(fp)
    
    tot_frames, tot_time = get_total_time(meta_info)
    
    # randomly assign the details
    calculate_partitions(meta_info, tot_time)
    
    
    
