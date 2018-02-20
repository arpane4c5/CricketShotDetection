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
                    labels = getBoundaryValuesForVideo(os.path.join(sub_src_path, vid))
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

    
# create labels for single video, params are srcVideo. Returns list of tuples like
# (preCutFrameNo, postCutFrameNo) on for Cut boundaries
def getBoundaryValuesForVideo(srcVideo):
    
    print "Begin New Video : "+srcVideo
    cap = cv2.VideoCapture(srcVideo)
    sbdLabels = []
    i = 0

    if not cap.isOpened():
        print "Could not open the video file !! Abort !!"
        return None
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret==True:
            cv2.imshow("Prev",frame)
            ret, frame = cap.read()
            if ret==True:
                print "Prev Frame : "+str(i)+" ## Next Frame : "+str(i+1)+" / "+str(length) 
                cv2.imshow("Next", frame)
            else:
                print "Next frame is NULL"
            direction = waitTillEscPressed()
            if direction == 1:
                sbdLabels.append(False)
                i +=1
            elif direction == 2:
                sbdLabels.append(True)
                i +=1
                #sbdLabels.append((i-1,i))
            else:
                i -=1
                sbdLabels.pop()            
        else:
            break

    sbd = [(idx,idx+1) for idx,isBoundary in enumerate(sbdLabels) if isBoundary]
    print("No. of boundaries in video : "+str(sum(sbdLabels)))
    #print(sbdLabels)
    print("Total no of frames traversed : ")
    print(i)
    cap.release()
    cv2.destroyAllWindows()
    
    return sbd

def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(0)==27:
            print("Esc Pressed. Move Forward without labeling.")
            return 1
        # For moving back
        elif cv2.waitKey(0)==98:
            print("'b' pressed. Move Back.")
            return 0
        elif cv2.waitKey(0)==121:
            print("'y' pressed. Label and Move Forward.")
            return 2


if __name__=='__main__':
    srcVideoFolder = "/home/hadoop/VisionWorkspace/Cricket/dataset_25_fps_test_set_1"
    
    destFolder = "/home/hadoop/VisionWorkspace/Cricket/group1/sbd"
    create_labels(srcVideoFolder, destFolder)
    
    