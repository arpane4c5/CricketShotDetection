# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:54:51 2016

Description: Verify the Ground Truth and check the shot segments

@author: Bloodhound
"""

import cv2
import os
import json

# function to iterate over the shot segments in the subfolders and manually verify
# shot labels for them.
def verify_labels(srcFolderPath, srcLabelsFolder, stop='all'):
    # iterate over the subfolders in srcFolderPath and extract for each video 
    sfp_lst = sorted(os.listdir(srcLabelsFolder))
    
    traversed_tot = 0
    for sf in sfp_lst:
        traversed = 0
        sub_src_path = os.path.join(srcFolderPath, sf)
        sub_labs_path = os.path.join(srcLabelsFolder, sf)
        if os.path.isdir(sub_labs_path) and os.path.isdir(sub_src_path):
            # iterate over the video files inside the directory sf
            vfiles = sorted(os.listdir(sub_labs_path))
            # Iterate over json files in subfolder
            for vlab in vfiles:
                label_file = os.path.join(sub_labs_path,vlab)
                # read segment labels into dictionary
                with open(label_file, 'r') as fp:
                    shots_dict = json.load(fp)
                vid_file = os.path.join(srcFolderPath, shots_dict.keys()[0])
                if os.path.isfile(vid_file):
                    viewShotSegmentsForVideo(vid_file, shots_dict)
                    # save at the destination, if extracted successfully
                    traversed += 1
                    print "Done "+str(traversed_tot+traversed)+" : "+sf+"/"+vid_file
                        
                    # to stop after successful traversal of 2 videos, if stop != 'all'
                    if stop != 'all' and traversed == stop:
                        break
            traversed_tot += traversed
                    
    print "No. of files written to destination : "+str(traversed_tot)
    if traversed_tot == 0:
        print "Check the structure of the dataset folders !!"
    return traversed_tot


# create cricket shot labels for single video, params are srcVideo. Returns list of
# tuples like (starting_frame, ending_frame)
def viewShotSegmentsForVideo(srcVideo, shots_dict):
    
    cap = cv2.VideoCapture(srcVideo)
    shotLabels = shots_dict[shots_dict.keys()[0]]
    i = 0
    print "Video : "+srcVideo
    print "#Shot Segments : "+str(len(shotLabels))
    if not cap.isOpened():
        print "Could not open the video file !! Abort !!"
        return None
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for shot in shotLabels:
        # shot is a tuple like (start, end)
        i = shot[0]
        while cap.isOpened() and i<=shot[1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Frame", frame)
                print "Shot : "+ str(shot)+ " ## Count : "+str(shot[1]-shot[0]+1)+ \
                " ## i : "+ str(i)
                direction = waitTillEscPressed()
                if direction == 1:
                    i +=1
                elif direction == 0:
                    i -=1
                elif direction == 2:
                    break
                elif direction == 3:
                    break
            else:
                print "Next frame is NULL"
        if direction==3:    # next video
            break

    cap.release()
    cv2.destroyAllWindows()
    return 

# get the starting and ending indices from the list of values.
def getListOfShots(shotLabels):
    shots = []
    start, end = -1, -1
    for i,isShot in enumerate(shotLabels):
        if isShot:
            if start<0:     # First True after a sequence of False
                start = i+1   
        else:
            if start>0:     # First false after a sequence of True
                end = i
                shots.append((start,end))
                start,end = -1,-1
    return shots

def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(0)==27:
            print("Esc Pressed. Move Forward.")
            return 1
        # For moving back
        elif cv2.waitKey(0)==98:
            print("'b' pressed. Move Back.")
            return 0
        # move to next shot segment
        elif cv2.waitKey(0)==110:
            print("'n' pressed. Move to next shot.")
            return 2
        # move to next video
        elif cv2.waitKey(0)==112:
            print("'p' pressed. Move to next video.")
            return 3


if __name__=='__main__':
    srcVideoFolder = "/home/hadoop/VisionWorkspace/Cricket/ToBeDeleted"
    srcLabelsFolder = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/dataset_25_fps_val_set_labels/corrected_shots"
    verify_labels(srcVideoFolder, srcLabelsFolder)
    
    