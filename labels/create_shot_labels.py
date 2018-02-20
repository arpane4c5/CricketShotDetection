# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:54:51 2016

Description: Create Ground Truth and write to json file for SBD

@author: Bloodhound
"""

import cv2
import os
import json

# function to iterate over the videos in the subfolders and create shot boundary
# labels for them.
def create_labels(srcFolderPath, destFolderPath, stop='all'):
    # iterate over the subfolders in srcFolderPath and extract for each video 
    sfp_lst = sorted(os.listdir(srcFolderPath))
    
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
            vfiles = sorted(os.listdir(sub_src_path))
            
            for vid in vfiles:
                if os.path.isfile(os.path.join(sub_dest_path, vid.rsplit('.',1)[0]+".json")):
                    continue
                
                if os.path.isfile(os.path.join(sub_src_path, vid)) and vid.rsplit('.', 1)[1] in {'avi', 'mp4'}:
                    labels = getShotLabelsForVideo(os.path.join(sub_src_path, vid))
                    # save at the destination, if extracted successfully
                    if not labels is None:
                        destFile = os.path.join(sub_dest_path, vid.rsplit('.',1)[0])+".json"
                        with open(destFile, "w") as fp:
                            json.dump({sf+'/'+vid : labels}, fp)
                        traversed += 1
                        print "Done "+str(traversed_tot+traversed)+" : "+sf+"/"+vid
                    else:
                        print "Labels file not created !!"
                        
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
def getShotLabelsForVideo(srcVideo):
    
    cap = cv2.VideoCapture(srcVideo)
    shotLabels = []
    i = 0
    isShot = False
    if not cap.isOpened():
        print "Could not open the video file !! Abort !!"
        return None
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened() and i<=(length+2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Prev",frame)
            ret, frame = cap.read()
            if ret:
                print "Prev : "+str(i)+" ## Next : "+str(i+1)+" ## Shot : " \
                                   +str(isShot)+"  / "+str(length) 
                cv2.imshow("Next", frame)
            else:
                print "Next frame is NULL"
            direction = waitTillEscPressed()
            if direction == 1:
                i +=1
                shotLabels.append(isShot)
            elif direction == 0:
                i -=1
                shotLabels.pop()
            elif direction == 2:
                if not isShot:
                    isShot = True
                else:
                    print "Shot already started. Press 'b' to move back and edit."
                #shotLabels.append(isShot)
            elif direction == 3:
                if isShot:
                    isShot = False
                else:
                    print "Shot not started yet. Press 'b' to move back and edit."
                
        else:
            break
    
    shots_lst = getListOfShots(shotLabels)

    print("No. of cricket shots in video : "+str(len(shots_lst)))
    print(shots_lst)
    print("Total no of frames traversed : ")
    print(i)
    cap.release()
    cv2.destroyAllWindows()
    
    return shots_lst

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
        # start of shot
        elif cv2.waitKey(0)==115:
            print("'s' pressed. Start of shot.")
            return 2
        # end of shot
        elif cv2.waitKey(0)==102:
            print("'f' pressed. End of shot.")
            return 3


if __name__=='__main__':
    srcVideoFolder = "/home/hadoop/VisionWorkspace/Cricket/dataset_25_fps_test_set_1"
    destFolder = "/home/hadoop/VisionWorkspace/Cricket/group1/create_shot"
    create_labels(srcVideoFolder, destFolder)
    
    