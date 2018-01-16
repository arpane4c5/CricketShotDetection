# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 20:21:16 2017

@author: Arpan

Description: Shot Selection functions dump
"""
import cv2
import numpy as np


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
    cv2.imshow("Target", tarImg)
    waitTillEscPressed()
    cv2.destroyAllWindows()
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


### methods defined in shot_extraction.py file

#
#def extract_shot(srcVideo):
#    cap = cv2.VideoCapture(srcVideo)
#    # if the VideoCapture object is not opened then exit
#    if not cap.isOpened():
#        import sys
#        print "Error in opening video File !!"
#        sys.exit(0)
#
#    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#    fps = cap.get(cv2.CAP_PROP_FPS)
#    frameCounter = 0
#    #fgbg = cv2.BackgroundSubtractorMOG2()
#    #lower = (29, 86, 6)
#    #upper = (64, 255, 255)      # use range-detector script of imutils
#    
#    ret, prev_frame = cap.read()
#    prev_frame = prev_frame[50:310,100:540]   # Cropping
#    # convert frame to GRAYSCALE
#    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#    # detect zooming when VideoCapture object is created
#    while(cap.isOpened()):
#        ret, frame = cap.read()
#        frame = frame[50:310,100:540]       # Cropping
#
#        if ret:
#            # Do the necessary processing here
#            #frame = imutils.resize(frame, width=600)
#                
#            #fgmask = fgbg.apply(frame)
#            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
################################################################################
#            # Optical Flow             
#            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
#            
#            flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#            
#            vis_bgr = draw_flow_bgr(flow, frame)
#            vis_vecs = draw_flow(prev_frame, flow, step=8)
#            cv2.imshow('Flow Vis', vis_bgr)
#            cv2.imshow('Flow Vecs', vis_vecs)
#
#            
################################################################################ 
#             
#            # Sobel filter
#            curr_frame = np.float32(curr_frame) / 255.0
#            gx = cv2.Sobel(curr_frame, cv2.CV_32F, 1, 0, ksize=3)
#            gy = cv2.Sobel(curr_frame, cv2.CV_32F, 0, 1, ksize=3)
#            print "Gx : "
#            display_mat_stats(gx)
#            ## Draw gradient BGR 
#            vis_grad_bgr = draw_flow_bgr(np.stack((gx, gy), axis=-1), frame)     #stack along last axis
#            cv2.imshow('Grad Vis', vis_grad_bgr)
#            # Python Calculate gradient magnitude and direction ( in degrees ) 
#            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
#            print "########## Frame No : "+str(frameCounter)+"  ##########"
#            print "Magnitude matrix : "
#            display_mat_stats(mag)
#            print "Angle matrix : "
#            display_mat_stats(angle)
#            
#            
#            # construct a mask for the defined 'color', then perform a series
#            # of dilations and erosions to remove any small blobs left in the 
#            # mask
#            #mask = cv2.inRange(hsv, lower, upper)
#            #mask = cv2.erode(mask, None, iterations=2)
#            #mask = cv2.dilate(mask, None, iterations=2)
#            cv2.imshow("Video", frame)
################################################################################            
#            # SIFT detector
#
#            sift = cv2.xfeatures2d.SIFT_create()
#            kp = sift.detect(frame,None)
#            img_sift=cv2.drawKeypoints(frame,kp,frame,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)           
#            cv2.imshow("SIFT", img_sift)
#            
#
################################################################################            
#            
#            #cv2.imshow('Cropped', frame[50:310,100:540])
#            #cv2.imshow("background", mask)
#            waitTillEscPressed()
#            #if cv2.waitKey(10)==27:
#            #    print("Esc Pressed")
#            #    break
#            frameCounter = frameCounter + 1
#            prev_frame = curr_frame
#        else:
#            break
#    cap.release()
#    cv2.destroyAllWindows()
#    return 


# Decide whether a video frame is a part of the positive example video shot
# Take features from the frame and predict using a trained model
def is_positive_shot_frame(frame):
    #import svm_model
#    # Sobel filter
#    frame = np.float32(frame) / 255.0
#    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
#    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
#    print "Gx : "
#    display_mat_stats(gx)
#    ## Draw gradient BGR 
#    vis_grad_bgr = draw_flow_bgr(np.stack((gx, gy), axis=-1), frame)     #stack along last axis
#    cv2.imshow('Grad Vis', vis_grad_bgr)
#    # Python Calculate gradient magnitude and direction ( in degrees ) 
#    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
#    print "Magnitude matrix : "
#    display_mat_stats(mag)
#    print "Angle matrix : "
#    display_mat_stats(angle)
    
    # Extract features of the frame (HOG etc) and
    # Predict using the trained SVM model 
    r = np.random.randint(2)    # coin toss, bernoulli RV
    if r==1:
        return True
    else:
        return False


# Frame Based Identification (FBI) using HOG descriptor
# Extract the shot from single video
# Inputs: srcVideoFolder and srcVideo define the complete path of video
#       cuts_list --> list of frame nos where boundary is predicted (postFNum)
def extract_shot_from_video(srcVideoFolder, srcVideo, cuts_list):
    cap = cv2.VideoCapture(os.path.join(srcVideoFolder,srcVideo))
    # if the VideoCapture object is not opened then exit
    if not cap.isOpened():
        import sys
        print "Error in opening video File !!"
        sys.exit(0)

    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_no_of_frames = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))
    start_frame = -1
    end_frame = -1
    cam1_flag = False   # True is for positive class of frame
    cam2_flag = False   # True if first frame of camera 2 is detected when
                        # shot is played, checked only after camera1 flag is true
    
    print "Video :: "+srcVideo
    print "Dim :: "+str(dimensions)+"  #### FPS :: "+str(fps)
    # see if it matches 360x640 and ~25 FPS
    no_of_cuts = len(cuts_list)
    
    for i in range(no_of_cuts):
        cut_pos = cuts_list[i]
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, cut_pos)
        ret, frame = cap.read()
        
        if ret:
            if not cam1_flag:
                if cam2_flag and not start_frame==-1:
                    end_frame = cut_pos
                    # call write_shot
                    start_frame = end_frame = -1
                    cam2_flag = False
                    
                cam1_flag = predict_frame_class_cam1(frame)
                if cam1_flag:
                    start_frame = cut_pos
                    continue
            else:
                cam1_flag = False
                cam2_flag = predict_frame_class_cam2(frame)
                if cam2_flag:
                    # check for last cut value
                    continue
                else:
                    end_frame = cut_pos
                    # call write_shot
                    start_frame = end_frame = -1
                    
                    
                
#    
        
        #writeShotToFile(srcVideo, start_index, stop_index, outFileName)
#    ret, prev_frame = cap.read()
#    #prev_frame = prev_frame[50:310,100:540]   # Cropping
#    # convert frame to GRAYSCALE
#    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#    
#    # If frame is not of a const dimension 360x640 then scale
#    # frame = cv2.resize(frame, (640, 360), intepolation = cv2.INTER_CUBIC)
#    # check FPS also, it should be ~25
#    while(cap.isOpened()):
#        ret, frame = cap.read()
#
#        if ret:
#            # Do the necessary processing here
#            #frame = imutils.resize(frame, width=600)
#            
#            #fgmask = fgbg.apply(frame)
#            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
################################################################################
#            # decide whether frame is part of shot or not
#            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
#
################################################################################ 
#            
################################################################################ 
#             
#            frame_res = is_positive_shot_frame(curr_frame)
#            
#            # Append value to a list and come up with a decision algo 
#            # for categorizing shot into a cricket shot
#            
#            cv2.imshow("Video", frame)            
#
################################################################################            
#            
#            #cv2.imshow('Cropped', frame[50:310,100:540])
#            #cv2.imshow("background", mask)
#            waitTillEscPressed()
#            #if cv2.waitKey(10)==27:
#            #    print("Esc Pressed")
#            #    break
#            frameCounter = frameCounter + 1
#            prev_frame = curr_frame
#        else:
#            break
    cap.release()
    cv2.destroyAllWindows()
    
    return 

