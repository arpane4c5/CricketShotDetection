#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 01:01:40 2017

@author: Arpan

Description: Script to track an object(preferably track a ball) in the cricket videos

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
 
      
#    while True:
#        # Read a new frame
#        ok, frame = video.read()
#        if not ok:
#            break

#        # Display result
#        cv2.imshow("Tracking", frame)
# 
#        # Exit if ESC pressed
#        k = cv2.waitKey(1) & 0xff
#        if k == 27 : break
#    
    

def track_in_video(srcVideo):

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
 
    if int(cv2.__version__.split('.')[1]) < 3:
        tracker = cv2.Tracker_create(tracker_type)
        print "Tracker initialized!!"
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    cap = cv2.VideoCapture(srcVideo)
    # if the VideoCapture object is not opened then exit
    if not cap.isOpened():
        import sys
        print "Error in opening video File !!"
        sys.exit(0)

    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameCounter = 0
    #fgbg = cv2.BackgroundSubtractorMOG2()
    #lower = (29, 86, 6)
    #upper = (64, 255, 255)      # use range-detector script of imutils
    
    ###########################################################################
    # Tracking a bounding box          
    # (x, y, w, h)
    bbox = (177, 115, 5, 5)  # get a localized area of interest
    cap.set(cv2.CAP_PROP_POS_FRAMES, 97)
    ret, prev_frame = cap.read()
    prev_frame = prev_frame[50:310,100:540]   # Cropping
    # convert frame to GRAYSCALE
    
    # detect zooming when VideoCapture object is created
 
    # Uncomment the line below to select a different bounding box
    #bbox = cv2.selectROI(prev_frame, False)
    #print bbox    
    
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(prev_frame, bbox)
    # Start timer
    #timer = cv2.getTickCount()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # Update tracker
    #ok, bbox = tracker.update(prev_frame)
 
    # Calculate Frames per second (FPS)
    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
    ###########################################################################
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = frame[50:310,100:540]       # Cropping

        if ret:
            # Do the necessary processing here
            #frame = imutils.resize(frame, width=600)
                
            #fgmask = fgbg.apply(frame)
            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
###############################################################################
            # Optical Flow             
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
            
            # Update Tracker
            ok, bbox = tracker.update(frame)
            
            flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            vis_bgr = draw_flow_bgr(flow, frame)
            vis_vecs = draw_flow(prev_frame, flow, step=8)
            cv2.imshow('Flow Vis', vis_bgr)
            cv2.imshow('Flow Vecs', vis_vecs)
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
############################################################################### 
             
            # Sobel filter
            curr_frame = np.float32(curr_frame) / 255.0
            gx = cv2.Sobel(curr_frame, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(curr_frame, cv2.CV_32F, 0, 1, ksize=3)
            print "Gx : "
            display_mat_stats(gx)
            ## Draw gradient BGR 
            vis_grad_bgr = draw_flow_bgr(np.stack((gx, gy), axis=-1), frame)     #stack along last axis
            cv2.imshow('Grad Vis', vis_grad_bgr)
            # Python Calculate gradient magnitude and direction ( in degrees ) 
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            print "########## Frame No : "+str(frameCounter)+"  ##########"
            print "Magnitude matrix : "
            display_mat_stats(mag)
            print "Angle matrix : "
            display_mat_stats(angle)
            
            
            # construct a mask for the defined 'color', then perform a series
            # of dilations and erosions to remove any small blobs left in the 
            # mask
            #mask = cv2.inRange(hsv, lower, upper)
            #mask = cv2.erode(mask, None, iterations=2)
            #mask = cv2.dilate(mask, None, iterations=2)
            cv2.imshow("Video", frame)
###############################################################################            
            # SIFT detector

            sift = cv2.xfeatures2d.SIFT_create()
            kp = sift.detect(frame,None)
            img_sift=cv2.drawKeypoints(frame,kp,frame,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)           
            cv2.imshow("SIFT", img_sift)
            

###############################################################################            
            
            #cv2.imshow('Cropped', frame[50:310,100:540])
            #cv2.imshow("background", mask)
            waitTillEscPressed()
            #if cv2.waitKey(10)==27:
            #    print("Esc Pressed")
            #    break
            frameCounter = frameCounter + 1
            prev_frame = curr_frame
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return 

def display_mat_stats(mat):
    print "Shape : "+str(mat.shape)+"  ::  Type : "+str(type(mat))
    print "Sum of values : "+str(np.sum(mat))
    print "Mean : "+str(np.mean(mat))+"  ::  SD : "+str(np.std(mat))  
    #print mat
    return


# draw the OF field on image, with grids, decrease step for finer grid
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_flow_bgr(flow, sample_frame):
    hsv = np.zeros_like(sample_frame)
    #print "hsv_shape : "+str(hsv.shape)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def waitTillEscPressed():
    while(True):
        if cv2.waitKey(10)==27:
            print("Esc Pressed")
            return
            

if __name__=='__main__':
    srcVideoFolder = "/home/hadoop/VisionWorkspace/VideoData/ICC WT20"
    srcVideosList = os.listdir(srcVideoFolder)
    print srcVideosList
    
    track_in_video(os.path.join(srcVideoFolder, srcVideosList[1]))
    #zoom_detect(os.path.join(srcVideoFolder, srcVideosList[0]))
    #check_sift("/home/hadoop/VisionWorkspace/ActivityProjPy/frames_col/f99.jpg", "/home/hadoop/VisionWorkspace/ActivityProjPy/frames_col/f112.jpg")
