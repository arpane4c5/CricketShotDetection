#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 12:44:03 2018

@author: Arpan

@Description: Utils file to extract chi-squared histogram difference features from videos

Feature : Chi-squared Histogram Difference features: RGB / Grayscale

"""

import os
import numpy as np
import cv2
#np.seterr(divide='ignore', invalid='ignore')
#import pandas as pd
#import SBD_xml_operations as lab_xml    
    
# function to extract the features from a list of videos
# Params: srcFolderPath = path to subfolders which contains the videos
# destFolderPath: path to store the hist-diff values in .bin files
# color: tuple specifying the channels (('g') for grayscale, ('b','g','r') for RGB)
# stop: to traverse 'stop' no of files in each subdirectory.
# Return: traversed: no of videos traversed successfully
def extract_wt_chi_sq_diff_vids(srcFolderPath, destFolderPath, color=('b','g','r'), stop='all'):
    # iterate over the subfolders in srcFolderPath and extract for each video 
    sfp_lst = os.listdir(srcFolderPath)
    
    traversed_tot = 0
    for sf in sfp_lst:
        traversed = 0
        sub_src_path = os.path.join(srcFolderPath, sf)
        sub_dest_path = os.path.join(destFolderPath, sf)
        if os.path.isdir(sub_src_path):
            # create destination path to store the files
            if not os.path.exists(sub_dest_path):
                os.makedirs(sub_dest_path)
            
            # iterate over the video files inside the directory sf
            vfiles = os.listdir(sub_src_path)
            for vid in vfiles:
                if os.path.isfile(os.path.join(sub_src_path, vid)) and vid.rsplit('.', 1)[1] in {'avi', 'mp4'}:
                    vdiffs = getWtChiSqHistogramOfVideo(os.path.join(sub_src_path, vid), color)
                    # save at the destination, if extracted successfully
                    if not vdiffs is None:
                        outfile = file(os.path.join(sub_dest_path, vid.rsplit('.', 1)[0])+".bin", "wb")
                        np.save(outfile, vdiffs)
                        outfile.close()
                        traversed += 1
                        print "Done "+str(traversed_tot+traversed)+" : "+sf+"/"+vid
                        
                    # to stop after successful traversal of 2 videos, if stop != 'all'
                    if stop != 'all' and traversed == stop:
                        break
            traversed_tot += traversed
                    
    print "No. of files written to destination : "+str(traversed_tot)
    if traversed_tot == 0:
        print "Check the structure of the dataset folders !!"
    
    return traversed_tot


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

    
if __name__=='__main__':
    srcPath = '/home/arpan/DATA_Drive/Cricket/dataset_25_fps'
    #srcPath = '/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset'
    destPath = "/home/arpan/VisionWorkspace/shot_detection/extracted_features/wtChiSq_diff_bgr_ds_25_fps"
    #destPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/features/test_chi_sq"
    extract_wt_chi_sq_diff_vids(srcPath, destPath, stop='all')