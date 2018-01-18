#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:00:08 2017

@author: hadoop
"""

import cv2
import os

# This path contains 4 subfolders : youtube, hotstar_converted, ipl2017, cpl2015
DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps/"
notOpened, opened = 0, 0

def write_meta_data(srcVid):
    global notOpened, opened
    cap = cv2.VideoCapture(srcVid)
    if not cap.isOpened():
        notOpened += 1
        print "Not Opened !!! "+srcVid
        return
    dimensions = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    ret, frame = cap.read()
    
    opened+=1
    print "{} File: {}".format(opened, srcVid)
    print "\t Dim: {}, FPS: {}, #Frames: {}, ret: {}" \
                 .format(dimensions, fps, nFrames, ret)
    cap.release()
    return


if __name__=="__main__":
    
    subfolders = os.listdir(DATASET)
    
    for sf in subfolders:
        for fl in os.listdir(os.path.join(DATASET, sf)):    # iterate over files
            write_meta_data(os.path.join(DATASET,sf, fl))
            
    print "Files Opened: {}, Not Opened: {}".format(opened, notOpened)