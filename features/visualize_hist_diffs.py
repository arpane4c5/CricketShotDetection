#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:44:25 2018

@author: Arpan

@Description : Visualize histogram difference features given the path of the folder where
the features were extracted.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import SBD_xml_operations as lab_xml    
    
# function to take the features from folder videos one at a time and visualize them
# Params: srcFolderPath = path to subfolders which contains the extracted features
# destFolderPath: path to store the hist-diff visualizations in .png files 
# color: tuple specifying the channels (('g') for grayscale, ('b','g','r') for RGB)
# stop: to traversel 'stop' no of files in each subdirectory.
# Return: traversed: no of videos traversed successfully
def vis_hist_diff_features(srcFolderPath, destFolderPath, stop='all'):
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
            
            # iterate over the bin files inside the directory sf
            binfiles = os.listdir(sub_src_path)
            for bfile in binfiles:
                if os.path.isfile(os.path.join(sub_src_path, bfile)) and bfile.rsplit('.', 1)[1] in ['bin','npy','pkl','dat']:
                    ret = visualizeFeature(sub_src_path, sub_dest_path, bfile)
                    # save at the destination, if extracted successfully
                    if ret:
                        print "Done : "+sf+"/"+bfile
                        traversed += 1
                        
                    # to stop after successful traversal of 2 videos, if stop != 'all'
                    if stop != 'all' and traversed == stop:
                        break
            traversed_tot += traversed
                    
    print "No. of files written to destination : "+str(traversed_tot)
    if traversed_tot == 0:
        print "Check the structure of the dataset folders !!"
    
    return traversed_tot


# function to plot the graph of the feature and save it in the 
def visualizeFeature(srcBinFilePath, destBinFilePath, filename):
    # load the binary file
    feat = np.load(os.path.join(srcBinFilePath, filename))
    file_prefix = filename.rsplit('.', 1)[0]
    nchannels = feat.shape[1]
    if nchannels == 1:  # For grayscale
        color = ('c')
    elif nchannels == 3:    # For BGR
        color = ('b', 'g', 'r')
    else:       # For any other number of channels
        color = range(1, nchannels+1)
    plt.close("all")
    # draw the line graph of the diffs signal
    for i,col in enumerate(color):
        fig = plt.figure(i)
        plt.title("Summed Absolute Histogram Differences of frames")
        plt.xlabel("Frame No.")
        plt.ylabel("Summed Absolute Hist. Diff.")
        ax = fig.add_subplot(111)
        plot_points = feat[:,i]
        plt.plot(plot_points, color=col, label=str(i))
        #print "Reached here!! "
        # loop over only the peaks above threshold
        peaks = [(x,y) for (x,y) in zip(range(len(plot_points)),plot_points) if y>100000]
        for ind,(x,y) in enumerate(peaks):
            ax.annotate('(%s,s'%x+str(ind)+')', xy=(x,y), textcoords='data')
        plt.grid()
        #plt.plot(feat)
        plt.savefig(os.path.join(destBinFilePath, file_prefix+"_"+str(col)+".png"), bbox_inches='tight')    
        
    return True



if __name__=='__main__':
    #srcPath = '/opt/datasets/KTH'
    #srcPath = '/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset'
    #destPath = "/home/arpan/DATA_Drive/Cricket/test_extract_hist"
    #destPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/features/test_feats"
    srcPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/features/test_hist_diffs"
    destPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/features/vis_test_hist_diffs"
    vis_hist_diff_features(srcPath, destPath, stop=3)
