#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 19:57:55 2017

@author: Arpan
Description: Extract frames cam1 and cam2 for positive egs. and negative egs. 
and save in respective folders.
    
"""

import cv2
import numpy as np
import os

# create labels for single video, params are srcVideo and dest frames filepath
def extract_sample_frames(srcVideo, destFilePath, count):
    cap = cv2.VideoCapture(srcVideo)    
    videoLabels = []
    i = 0
    #xmlFileName = os.path.join(destFolder, "test_"+srcVideoFile.split(".")[0]+".xml")

    if not cap.isOpened():
        print "Could not open the video file !! Abort !!"
        return False
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # define the root tag
#    root = etree.Element('refSeg')
#    tree = etree.ElementTree(root)
#    root.set('src',srcVideo)
#    root.set('creationMethod','MANUAL')
#    root.set('totalFNum', str(length))
    # create child tag
    if not os.path.exists(destFilePath):
        os.makedirs(destFilePath)
    
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret==True:
            cv2.imshow("Frame",frame)
            #ret, frame = cap.read()
            print "Frame : "+str(i)
            direction = waitTillEscPressed()
            if direction == 1:
                i = i+1
                #videoLabels.append(0)
            elif direction == 2:
                i = i+1
                filename = os.path.join(destFilePath, "pos_"+str(count)+"_"+str(i-1)+".jpg")
                cv2.imwrite(filename, frame)
            elif direction == 3:
                i = i+1
                filename = os.path.join(destFilePath, "neg_"+str(count)+"_"+str(i-1)+".jpg")
                cv2.imwrite(filename, frame)
            elif direction == 4:
                break
            else:
                i = i-1
                #videoLabels.pop()
        else:
            break

    #print("Video Frame Labels with len : "+str(len(videoLabels)))
    #print(videoLabels)
    print("Total no of frames : ")
    print(i)
    cap.release()
    cv2.destroyAllWindows()
    
    return True


def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(10)==27:
            print("Esc Pressed. Move Forward without labeling.")
            return 1
        # For moving back
        elif cv2.waitKey(10)==98:
            print("'b' pressed. Move Back.")
            return 0
        elif cv2.waitKey(10)==121:
            print("'y' pressed. +ve Label and Move Forward.")
            return 2
        elif cv2.waitKey(10)==110:
            print("'n' pressed. -ve Label and Move Forward.")
            return 3
        elif cv2.waitKey(10)==112:
            print("'p' pressed. Escape CUrrect Video")
            return 4


def display_mat_stats(mat):
    print "Shape : "+str(mat.shape)+"  ::  Type : "+str(type(mat))
    print "Sum of values : "+str(np.sum(mat))
    print "Mean : "+str(np.mean(mat))+"  ::  SD : "+str(np.std(mat))  
    #print mat
    return

# check SIFT descriptor
def check_sift(srcImage, targetImage):
    srcImg = cv2.imread(srcImage)
    srcImg = srcImg[50:310,100:540]
    tarImg = cv2.imread(targetImage)
    tarImg = tarImg[50:310,100:540]
    srcImg_gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(srcImg_gray,None)
    print srcImg.shape
    img_sift=cv2.drawKeypoints(srcImg,kp,tarImg,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("SIFT", img_sift)
    cv2.imshow("Original", srcImg)
    cv2.imshow("GRAY", srcImg_gray)
    cv2.imshow("Traget", tarImg)
    waitTillEscPressed()
    cv2.destroyAllWindows()
    return

if __name__=='__main__':
    #srcVideoFolder = "D:/LNMIIT PhD/Thesis Project/ICC WT20"
    srcVideoFolder = "/home/hadoop/VisionWorkspace/VideoData/ICC WT20"
    srcVideosList = sorted(os.listdir(srcVideoFolder))
    print srcVideosList
    
    #destFolder = "D:/Workspaces/PythonOpenCV/ActivityProjPy/ExtractFrames/cam2_frames"
    destFolder = "/home/hadoop/VisionWorkspace/Cricket/scripts/camera_models/cam2_frames"
    #Execute for all the videos in the folder
    for count,filename in enumerate(srcVideosList):
        extract_sample_frames(os.path.join(srcVideoFolder, srcVideosList[count]), destFolder, count)
    #zoom_detect(os.path.join(srcVideoFolder, srcVideosList[0]))
    #check_sift("/home/hadoop/VisionWorkspace/ActivityProjPy/frames_col/f99.jpg", "/home/hadoop/VisionWorkspace/ActivityProjPy/frames_col/f112.jpg")
