#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sept 8 01:34:25 2018
@author: Arpan
@Description: Utils file to extract c3d features from dataset videos and dump 
to disk.
Feature : C3D features extracted from the videos, taken pretrained model (on Sports-1M)
and extracting fc7 layer feature vectors. (4096 size)
"""

import torch
import os
import numpy as np
import cv2
import time
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
from joblib import Parallel, delayed

def extract_c3d_all(model, srcFolderPath, destFolderPath, njobs=1, batch=10, \
                    onGPU=True, depth=16, stop='all'):
    """
    Function to extract the features from a list of videos
    
    Parameters:
    ------
    model: C3D
        C3D model loaded with pretrained weights (on Sports-1M)
    srcFolderPath: str
        path to folder which contains the videos
    destFolderPath: str
        path to store the optical flow values in .bin files
    onGPU: boolean
        True enables a serial extraction by sending model and data to GPU,
        False enables a parallel extraction on the different CPU cores.
    depth: int
        No. of frames (min 16) taken from video and fed to C3D at a time.
    stop: str
        to traversel 'stop' no of files in each subdirectory.
    
    Returns: 
    ------
    traversed: int
        no of videos traversed successfully
    """    
    # iterate over the subfolders in srcFolderPath and extract for each video 
    sfp_lst = os.listdir(srcFolderPath)
    infiles, outfiles, nFrames = [], [], []
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
                    infiles.append(os.path.join(sub_src_path, vid))
                    outfiles.append(os.path.join(sub_dest_path, vid.rsplit('.',1)[0]+".npy"))
                    nFrames.append(getTotalFramesVid(os.path.join(sub_src_path, vid)))
                    traversed += 1
#                   print "Done "+str(traversed_tot+traversed)+" : "+sf+"/"+vid                    
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
    
    ###########################################################################
    if onGPU:
        # Serial Implementation (For GPU based extraction)
        for i in range(nrows):
            st = time.time()
            feat = getC3DFrameFeats(model, filenames_df['infiles'][i], onGPU, depth)
            # save the feature to disk
            if feat is not None:
                np.save(filenames_df['outfiles'][i], feat)
                print "Written "+str(i)+" : "+filenames_df['outfiles'][i]
                
            e = time.time()
            print "Execution Time : "+str(e-st)
    
    else:    
        #feat = getC3DFrameFeats(model, filenames_df['infiles'][0], onGPU, depth)
        # Parallel version (For CPU based extraction)
        for i in range(nrows/batch):
            
            batch_diffs = Parallel(n_jobs=njobs)(delayed(getC3DFrameFeats) \
                        (model, filenames_df['infiles'][i*batch+j], onGPU, depth) \
                        for j in range(batch))
            print "i = "+str(i)
            # Writing the diffs in a serial manner
            for j in range(batch):
                if batch_diffs[j] is not None:
                    np.save(filenames_df['outfiles'][i*batch+j], batch_diffs[j])
                    print "Written "+str(i*batch+j+1)+" : "+ \
                                filenames_df['outfiles'][i*batch+j]
                
        # For last batch which may not be complete, extract serially
        last_batch_size = nrows - ((nrows/batch)*batch)
        if last_batch_size > 0:
            batch_diffs = Parallel(n_jobs=njobs)(delayed(getC3DFrameFeats) \
                        (model, filenames_df['infiles'][(nrows/batch)*batch+j], onGPU, depth) \
                        for j in range(last_batch_size)) 
            # Writing the diffs in a serial manner
            for j in range(last_batch_size):
                if batch_diffs[j] is not None:
                    np.save(filenames_df['outfiles'][(nrows/batch)*batch+j], batch_diffs[j])
                    print "Written "+str((nrows/batch)*batch+j+1)+" : "+ \
                                filenames_df['outfiles'][(nrows/batch)*batch+j]
    
    ###########################################################################
#    print len(batch_diffs)
    return traversed


def getTotalFramesVid(srcVideoPath):
    """
    Return the total number of frames in the video
    
    Parameters:
    ------
    srcVideoPath: str
        complete path of the source input video file
        
    Returns:
    ------
    total frames present in the given video file
    """
    cap = cv2.VideoCapture(srcVideoPath)
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return 0

    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return tot_frames    


def getC3DFrameFeats(model, srcVideoPath, onGPU, depth):
    """
    Function to read all the frames of the video and get sequence of features
    by passing 'depth' frames to C3D model, one batch at a time. 
    This function can be called parallely called based on the amount of 
    memory available.
    
    Parameters:
    ------
    model: model_c3d.C3D
        torch.nn.Module subclass for C3D network, defined in model_c3d.C3D
    srcVideoPath: str
        complete path of the src video folder
    depth: int
        no. of frames taken as input to the C3D model to generate a single 
        output vector. Min. is 16 (trained as such in paper)
        
    Returns:
    ------
    np.array of size (N-depth+1) x 4096 (N is the no. of frames in video.)
    """
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
    W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    features_current_file = []
    
    #ret, prev_frame = cap.read()
    assert cap.isOpened(), "Capture object does not return a frame!"
    #prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    X = []  # input, initially a list, after first 16 frames converted to ndarray
    # Iterate over the entire video to get the optical flow features.
    while(cap.isOpened()):
        
        ret, curr_frame = cap.read()    # H x W x C
        if not ret:
            break
        
        # resize to 180 X 320 and taking centre crop of 112 x 112
        curr_frame = cv2.resize(curr_frame, (W/2, H/2), cv2.INTER_AREA)
        (h, w) = curr_frame.shape[:2]
        # size is 112 x 112 x 3
        curr_frame = curr_frame[(h/2-56):(h/2+56), (w/2-56):(w/2+56), :]
        
        if frameCount < (depth-1):     # append to list till first 16 frames
            X.append(curr_frame)
        else:       # subsequent frames
            if type(X)==list:   # For exactly first 16 frames, convert to np.ndarray 
                X.append(curr_frame)
                X = np.stack(X)
                X = np.float32(X)
                X = torch.from_numpy(X)
                if onGPU:
                    X = X.cuda()
            else:   # sliding the window (taking 15 last frames and append next)
                    # Adding a new dimension and concat on first axis
                curr_frame = np.float32(curr_frame)
                curr_frame = torch.from_numpy(curr_frame)
                if onGPU:
                    curr_frame = curr_frame.cuda()
                #X = np.concatenate((X[1:], curr_frame[None, :]), axis=0)
                X = torch.cat([X[1:], curr_frame[None, :]])
        
            # TODO: Transpose once, and concat on first axis for subsequent frames
            # passing the matrix X to the C3D model
            # X is (depth, H, W, Ch)
            #input_mat = X.transpose(3, 0, 1, 2)     # ch, depth, H, W
            input_mat = X.permute(3, 0, 1, 2)       # transpose a 4D torch Tensor
            #input_mat = np.expand_dims(input_mat, axis=0)
            input_mat = input_mat.unsqueeze(0)      # expand dims on Tensor
            #input_mat = np.float32(input_mat)
            
            # Convert to Variable
            #input_mat = torch.from_numpy(input_mat)
            input_mat = Variable(input_mat)
            
            # get the prediction after passing the input to the C3D model
            prediction = model(input_mat)
            # convert to numpy vector
            prediction = prediction.data.cpu().numpy()
            features_current_file.append(prediction)
            
        frameCount +=1
        #print "{} / {}".format(frameCount, totalFrames)

    # When everything done, release the capture
    cap.release()
    #return features_current_file
    return np.array(features_current_file)      # convert to (N-depth+1) x 1 x 4096


if __name__=='__main__':

    # The srcPath should have subfolders that contain the videos.
    weights_path = "/home/hadoop/VisionWorkspace/Cricket/localization_rnn/c3d.pickle"
    srcPath = "/home/hadoop/VisionWorkspace/Cricket/dataset_25_fps"
    destPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/extracted_features/c3d_feats_16"
    if not os.path.exists(srcPath):
        srcPath = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps"
        destPath = "/home/arpan/VisionWorkspace/shot_detection/extracted_features/c3d_feats_16"
        weights_path = "/home/arpan/VisionWorkspace/localization_rnn/c3d.pickle"
    
    onGPU = True    # Flag True if we want a GPU extract (Serial),
    # False if we want a parallel extraction on the CPU cores.
    
    batch = 5  # No. of videos in a single batch
    njobs = 1   # No. of threads    
    
    # create a model 
    import model_c3d as c3d
    
    model = c3d.C3D()
    
    ###########################################################################
    
    # get network pretrained model
    model.load_state_dict(torch.load(weights_path))
    if onGPU:
        model.cuda()
        
    model.eval()
    
    print "Using the GPU : "+str(onGPU)
    start = time.time()
    nfiles = extract_c3d_all(model, srcPath, destPath, njobs=njobs, \
                             batch=batch, onGPU=onGPU, depth=16, stop=2)
    end = time.time()
    print "Total no. of files traversed : "+str(nfiles)
    print "Total execution time : "+str(end-start)
    
    ###########################################################################
    # Results:
    # Data kept on the GPU and the frames are sent to the GPU and appended by 
    # sliding the window of size 16 on the GPU. The model is saved on the GPU
    # initially and a forward pass through the network gives the FC7 feature vectors.
    #
    # On GPU: Serial Execution time(26 vids) : 14497.0 sec
    #
    # Parallel Implementation: 5 cores 5 batch size
    # Execution time (26 vids) : 43924.76 sec