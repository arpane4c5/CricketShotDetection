#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 12:44:03 2018

@author: Arpan

@Description: Utils file to extract chi-squared histogram difference features from videos.
The extraction is parallelized over the different cores. #Jobs(Threads) = 15 
and #Batches = 50. A batch of 50 videos is finished parallely and written to disk.

Feature : Chi-squared Histogram Difference features: RGB / Grayscale

"""

import os
import numpy as np
import cv2
import pandas as pd
import time
from joblib import Parallel, delayed
np.seterr(divide='ignore', invalid='ignore') 
    
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
    infiles, outfiles, nFrames = [], [], []    
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
                    infiles.append(os.path.join(sub_src_path, vid))
                    outfiles.append(os.path.join(sub_dest_path, vid.rsplit('.',1)[0]+".bin"))
                    nFrames.append(getTotalFramesVid(os.path.join(sub_src_path, vid)))             
                    
                    
                    traversed += 1
                    
                        
                    # to stop after successful traversal of 2 videos, if stop != 'all'
                    if stop != 'all' and traversed == stop:
                        break
            traversed_tot += traversed
                    
    print "No. of files to be written to destination : "+str(traversed_tot)
    if traversed_tot == 0:
        print "Check the structure of the dataset folders !!"
        return traversed_tot

    ###########################################################################
    #### Form the pandas Dataframe and parallelize over the files.
    filenames_df = pd.DataFrame({"infiles":infiles, "outfiles": outfiles, "nframes": nFrames})
    filenames_df = filenames_df.sort_values(["nframes"], ascending=[True]) #sort by nFrames
    filenames_df = filenames_df.reset_index(drop=True)  # reset row names(index).
    nrows = filenames_df.shape[0]
    batch = 50      # No. of videos in a batch.
    njobs = 10      # No. of threads.
    
    for i in range(nrows/batch):
        # 
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getWtChiSqHistogramOfVideo) \
                          (filenames_df['infiles'][i*batch+j], color) \
                          for j in range(batch))
        
        # Writing the diffs in a serial manner
        for j in range(batch):
            if batch_diffs[j] is not None:
                outfile = file(filenames_df['outfiles'][i*batch+j] , "wb")
                np.save(outfile, batch_diffs[j])
                outfile.close()
                print "Written "+str(i*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][i*batch+j]
            
    # For last batch which may not be complete, extract serially
    last_batch_size = nrows - ((nrows/batch)*batch)
    if last_batch_size > 0:
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getWtChiSqHistogramOfVideo) \
                              (filenames_df['infiles'][(nrows/batch)*batch+j], color) \
                              for j in range(last_batch_size)) 
        # Writing the diffs in a serial manner
        for j in range(last_batch_size):
            if batch_diffs[j] is not None:
                outfile = file(filenames_df['outfiles'][(nrows/batch)*batch+j] , "wb")
                np.save(outfile, batch_diffs[j])
                outfile.close()
                print "Written "+str((nrows/batch)*batch+j+1)+" : "+ \
                                    filenames_df['outfiles'][(nrows/batch)*batch+j]
    

# return the total number of frames in the video
def getTotalFramesVid(srcVideoPath):
    cap = cv2.VideoCapture(srcVideoPath)
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return 0

    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return tot_frames    


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
            
        else:
            break

    # When the video is done with, release the capture object
    cap.release()

    return diffs

    
if __name__=='__main__':
    srcPath = '/home/arpan/DATA_Drive/Cricket/dataset_25_fps'
    #srcPath = '/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset'
    destPath = "/home/arpan/VisionWorkspace/shot_detection/extracted_features/wtChiSq_diff_bgr_ds_25_fps_par"
    #destPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/features/test_chi_sq"
    start = time.time()
    extract_wt_chi_sq_diff_vids(srcPath, destPath, stop='all')
    end = time.time()
    print "Total execution time : "+str(end-start)