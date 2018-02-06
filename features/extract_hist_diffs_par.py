#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:44:25 2018

@author: Arpan

@Description: Utils file to extract histogram difference features from folder videos

Feature : Histogram Difference features: RGB / Grayscale

"""

import os
import numpy as np
import cv2
import time
import pandas as pd
from joblib import Parallel, delayed
#import SBD_xml_operations as lab_xml    
    
# function to extract the features from a list of videos
# Params: srcFolderPath = path to subfolders which contains the videos
# destFolderPath: path to store the hist-diff values in .bin files
# color: tuple specifying the channels (('g') for grayscale, ('b','g','r') for RGB)
# stop: to traversel 'stop' no of files in each subdirectory.
# Return: traversed: no of videos traversed successfully
def extract_hist_diff_vids(srcFolderPath, destFolderPath, color=('g'), stop='all'):
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
                    #vdiffs = getHistogramOfVideo(os.path.join(sub_src_path, vid), color)
                    # save at the destination, if extracted successfully
#                    if not vdiffs is None:
#                        outfile = file(os.path.join(sub_dest_path, vid.rsplit('.', 1)[0])+".bin", "wb")
#                        np.save(outfile, vdiffs)
#                        outfile.close()
                    traversed += 1
#                        print "Done "+str(traversed_tot+traversed)+" : "+sf+"/"+vid
                        
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
    filenames_df = filenames_df.sort_values(["nframes"], ascending=[True])
    filenames_df = filenames_df.reset_index(drop=True)
    nrows = filenames_df.shape[0]
    batch = 50  # No. of videos in a single batch
    njobs = 10   # No. of threads
    
    for i in range(nrows/batch):
        # 
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getHistogramOfVideo) \
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
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getHistogramOfVideo) \
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
    
    ###########################################################################
    return traversed_tot
#    for idx, vid in enumerate(vids_lst):
#        #get_hist_diff(os.path.join(DATASET, vid+'.avi'))
#        diffs = getHistogramOfVideo(os.path.join(DATASET, vid+'.avi'), "", 100)
#        #print "diffs : ",diffs
#        print "Done : " + str(idx+1)
#        hist_diff_all.append(diffs)
#        # save diff_hist to disk    
#        #outfile = file(os.path.join(destPath,"diff_hist.bin"), "wb")
#        #np.save(outfile, diffs)
#        #outfile.close()    
#        #break
#    return hist_diff_all

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

# function to get the L1 distances of histograms and plot the signal
# for getting the grayscale histogram differences, uncomment two lines
# Copied and editted from shot_detection.py script
# color=('b') : For grayscale, ('b','g','r') for RGB
def getHistogramOfVideo(srcVideoPath, color=('b')):
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None

#    # create destination folder if not created already
#    if not os.path.exists(destPath):
#        os.makedirs(destPath)
    
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #out = cv2.VideoWriter('outputImran.avi', fourcc, fps, dimensions, True)
    #print(out)
    frameCount = 0
    #color = ('b', 'g', 'r')     # defined for 3 channels
    prev_hist = np.zeros((256, len(color)))
    curr_hist = np.zeros((256, len(color)))
    diffs = np.zeros((1, len(color)))
    while(cap.isOpened()):
        # Capture frame by frame
        ret, frame = cap.read()
        # print(ret)
    
        if ret==True:
            # frame = cv2.flip(frame)
            frameCount = frameCount + 1
            
            # condition for converting frames to grayscale
            # color = ('b')
            if len(color) == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         
            
            for i,col in enumerate(color):
                # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
                curr_hist[:,i] = np.reshape(cv2.calcHist([frame], [i], None, [256], [0,256]), (256,))
            
            if frameCount > 1:
                # find the L1 distance of the current frame hist to previous frame hist 
                dist = np.sum(abs(curr_hist - prev_hist), axis=0)
                #diffs.append(dist)
                diffs = np.vstack([diffs, dist])
                #print("dist = ", type(dist), dist)           
                #print("diffs = ", type(diffs), diffs.shape)
                #waitTillEscPressed()
            np.copyto(prev_hist, curr_hist)        
            
            # Display the resulting frame
            # cv2.imshow('frame', gray)
            #if cv2.waitKey(10) == 27:
            #    print('Esc pressed')
            #    break
        else:
            break

    # When everything done, release the capture
    cap.release()
    
    return diffs


if __name__=='__main__':
    # The srcPath should have subfolders that contain the training, val, test videos.
    # The function iterates over the subfolders and videos inside that.
    # The destPath will be created and inside that directory structure similar 
    # to src path will be created, with binary files containing the features.
    srcPath = '/home/arpan/DATA_Drive/Cricket/dataset_25_fps'
    #srcPath = '/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset'
    destPath = "/home/arpan/VisionWorkspace/shot_detection/extracted_features/hist_diff_bgr_ds_25_fps"
    #destPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/features/test_hist_diffs"
    start = time.time()
    extract_hist_diff_vids(srcPath, destPath, color=('b','g','r'), stop='all')
    end = time.time()
    print "Total execution time : "+str(end-start)