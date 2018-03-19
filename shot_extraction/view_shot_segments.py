# -*- coding: utf-8 -*-
"""
Created on Wed Mar 4 13:54:51 2016

Description: View predicted shot segments.

@author: Bloodhound
"""

import cv2
import os
import json

# Local System Path
DATASET_PATH = "/home/hadoop/VisionWorkspace/Cricket/dataset_25_fps"
# function to iterate over the shot segments in the shotsFile and view predicted 
# shots
def view_segments(shotsFile, stop='all'):
    # read the shots from the json file
    with open(shotsFile, 'r') as fp:
        all_shots_dict = json.load(fp)
    
    traversed_tot = 0
    for k in all_shots_dict.keys():
        vshots_dict = {k:all_shots_dict[k]}
        vid_file = os.path.join(DATASET_PATH, k)
        if os.path.isfile(vid_file):
            viewShotSegmentsForVideo(vid_file, vshots_dict)
            traversed_tot +=1
            print "Done "+str(traversed_tot)+" : "+vid_file
            if stop != 'all' and traversed_tot == stop:
                break
    
    print "No. of files traversed : "+str(traversed_tot)


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
    shotsFile = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/segment_filt/cricShots_hdiffGray_naive_v1_filt60.json"
    view_segments(shotsFile, stop=4)
    
    