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
#import pandas as pd
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
                    vdiffs = getHistogramOfVideo(os.path.join(sub_src_path, vid), color)
                    # save at the destination, if extracted successfully
                    if not vdiffs is None:
                        outfile = file(os.path.join(sub_dest_path, vid.rsplit('.', 1)[0])+".bin", "wb")
                        np.save(outfile, vdiffs)
                        outfile.close()
                        print "Done : "+sf+"/"+vid
                        traversed += 1
                        
                    # to stop after successful traversal of 2 videos, if stop != 'all'
                    if stop != 'all' and traversed == stop:
                        break
            traversed_tot += traversed
                    
    print "No. of files written to destination : "+str(traversed_tot)
    if traversed_tot == 0:
        print "Check the structure of the dataset folders !!"
    
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
            
            ### write the flipped frame
            ###out.write(frame)
            ### write frame to file
            ## Uncomment following 3 lines to get the images of the video saved to dir
            #filename = os.path.join(destPath,'f'+str(frameCount)+'.jpg')
            #cv2.imwrite(filename, frame)
            #print('Frame written', frameCount)
            
            # Display the resulting frame
            # cv2.imshow('frame', gray)
            if cv2.waitKey(10) == 27:
                print('Esc pressed')
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    #out.release()
    #cv2.destroyAllWindows()
#    plt.close("all")
#    # draw the line graph of the diffs signal
#    for i,col in enumerate(color):
#        fig = plt.figure(i)
#        plt.title("Summed Absolute Histogram Differences of frames")
#        plt.xlabel("Frame No.")
#        plt.ylabel("Summed Absolute Hist. Diff.")
#        ax = fig.add_subplot(111)
#        plot_points = diffs[:,i]
#        plt.plot(plot_points, color=col, label=str(i))
#        print "Reached here!! "
#        # loop over only the peaks above threshold
#        peaks = [(x,y) for (x,y) in zip(range(len(plot_points)),plot_points) if y>th]
#        for ind,(x,y) in enumerate(peaks):
#            ax.annotate('(%s,s'%x+str(ind)+')', xy=(x,y), textcoords='data')
#        plt.grid()
        #plt.plot(diffs)
        #plt.savefig(os.path.join(destPath,"hist_"+str(col)+".png"), bbox_inches='tight')

    return diffs


if __name__=='__main__':
    srcPath = '/opt/datasets/KTH'
    #srcPath = '/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset'
    destPath = "/home/arpan/DATA_Drive/Cricket/test_extract_hist"
    #destPath = "/home/hadoop/VisionWorkspace/Cricket/scripts/features/test_feats"
    extract_hist_diff_vids(srcPath, destPath, stop=3)