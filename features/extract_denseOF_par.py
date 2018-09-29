#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat June 29 01:34:25 2018

@author: Arpan

@Description: Utils file to extract Farneback dense optical flow features 
from folder videos

Feature : Farneback Dense Optical Flow: Magnitudes and Angles

"""

import os
import numpy as np
import cv2
import time
import pandas as pd
from joblib import Parallel, delayed
    
# function to extract the features from a list of videos
# Params: srcFolderPath = path to subfolders which contains the videos
# destFolderPath: path to store the optical flow values in .bin files
# grid_size: distance between two neighbouring pixel optical flow values.
# stop: to traversel 'stop' no of files in each subdirectory.
# Return: traversed: no of videos traversed successfully
def extract_dense_OF_vids(srcFolderPath, destFolderPath, grid_size=20, stop='all'):
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
    batch = 5  # No. of videos in a single batch
    njobs = 3   # No. of threads
    
    for i in range(nrows/batch):
        # 
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getFarnebackOFVideo) \
                          (filenames_df['infiles'][i*batch+j], grid_size) \
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
        batch_diffs = Parallel(n_jobs=njobs)(delayed(getFarnebackOFVideo) \
                              (filenames_df['infiles'][(nrows/batch)*batch+j], grid_size) \
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

# function to get the Farneback dense optical flow features for the video file
# and sample magnitude and angle features with a distance of grid_size between 
# two neighbours.
# Copied and editted from shot_detection.py script
# color=('b') : For grayscale, ('b','g','r') for RGB
def getFarnebackOFVideo(srcVideoPath, grid_size):
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    features_current_file = []
    
    ret, prev_frame = cap.read()
    assert ret, "Capture object does not return a frame!"
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Iterate over the entire video to get the optical flow features.
    while(cap.isOpened()):
        frameCount +=1
        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # stack sliced arrays along the first axis (2, (360/grid), (640/grid))
        sliced_flow = np.stack(( mag[::grid_size, ::grid_size], \
                                ang[::grid_size, ::grid_size]), axis=0)
        
        #feature.append(sliced_flow[..., 0].ravel())
        #feature.append(sliced_flow[..., 1].ravel())
        #feature = np.array(feature)
        features_current_file.append(sliced_flow)
        prev_frame = curr_frame

    # When everything done, release the capture
    cap.release()
    #print "{}/{} frames in {}".format(frameCount, totalFrames, srcVideoPath)
    return features_current_file


if __name__=='__main__':
    # The srcPath should have subfolders that contain the training, val, test videos.
    # The function iterates over the subfolders and videos inside that.
    # The destPath will be created and inside that directory structure similar 
    # to src path will be created, with binary files containing the features.
    #srcPath = '/home/arpan/DATA_Drive/Cricket/dataset_25_fps'
    srcPath = '/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset'
    #destPath = "/home/arpan/VisionWorkspace/shot_detection/extracted_features/OF_ds_25_fps"
    destPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/features/test_hist_diffs"
    start = time.time()
    extract_dense_OF_vids(srcPath, destPath, grid_size=20, stop='all')
    end = time.time()
    print "Total execution time : "+str(end-start)