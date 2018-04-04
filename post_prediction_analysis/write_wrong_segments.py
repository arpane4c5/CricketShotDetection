# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:54:51 2016

Description: Extract the wrong predictions by comparing with ground truth.

@author: Bloodhound
"""

import cv2
import os
import json

# Local System Path
DATASET_PREFIX = "/home/hadoop/VisionWorkspace/Cricket/dataset_25_fps"
SUPPORTING_FILES_PATH = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files"
GT_SHOTS = "dataset_25_fps_val_set_labels/corrected_shots"
#GT_SHOTS = "dataset_25_fps_test_set_labels"

# function to check where TIoU < threshold for a specific video in the validation 
# set and write the segments for that video to the file.
# gt_dir: directory where the GT labels are kept in subfolders
# shotsFile: json file with predictions.
# destPath: new path where wrong segments will be written
def write_wrong_segments(gt_dir, shotsFile, destPath, thresh=0, write_flag=False):
    # read the shots from the json file
    with open(shotsFile, 'r') as fp:
        shots_dict = json.load(fp)
    wrong_gt_path = os.path.join(destPath, "gt")
    wrong_pred_path = os.path.join(destPath, "pred")
    # Create destination path, if it doesn't exist
    if not os.path.exists(wrong_gt_path):
        os.makedirs(wrong_gt_path)
    if not os.path.exists(wrong_pred_path):
        os.makedirs(wrong_pred_path)
    # Iterate over the gt files and for each calculate the TIoU; 
    # at the same time write the wrongly detected segments.
    sfp_lst = os.listdir(gt_dir)
    tot_tiou = 0
    tot_segments = 0
    traversed_tot = 0
    for sf in sfp_lst:
        traversed = 0
        sub_src_path = os.path.join(gt_dir, sf)
        if os.path.isdir(sub_src_path):
            # iterate over the json files inside the directory sf
            labelFiles = os.listdir(sub_src_path)
            for labfile in labelFiles:
                if os.path.isfile(os.path.join(sub_src_path, labfile)) and labfile.rsplit('.', 1)[1] in ['json']:
                    with open(os.path.join(sub_src_path, labfile), 'r') as fp:
                        vid_gt = json.load(fp)
                    vid_key = vid_gt.keys()[0]      # only one key in dict is saved
                    gt_list = vid_gt[vid_key]       # list of tuples [[preFNum, postFNum], ...]
                    
                    test_list = shots_dict[vid_key]
                    print "Done "+str(traversed_tot+traversed)+" : "+vid_key
                    # calculate tiou for the video vid_key
                    #vid_tiou = get_vid_tiou(gt_list, test_list)
                    vid_tiou = get_wrong_vid_segments(vid_gt, test_list, wrong_gt_path, \
                                                      wrong_pred_path, thresh, write_flag)
                    # vid_tiou weighted with no of ground truth segments
                    tot_tiou += (vid_tiou*len(gt_list))
                    tot_segments += len(gt_list)
                    traversed += 1
                    
            traversed_tot += traversed
            
    print "Total files traversed : "+str(traversed_tot)
    print "Total segments : " + str(tot_segments)
    print "Total_tiou (all vids) : " + str(tot_tiou)
    #print "Averaged TIoU : " + str(tot_tiou/traversed_tot)
    print "Weighted Averaged TIoU  : " + str(tot_tiou/tot_segments)
    
    return (tot_tiou/tot_segments)

# function to write wrong shot segments for one video
# vid_gt_dict: dictionary with {vid_name: list_of_gt_segments}
# test_list: list of predicted segments
# destGT: destination for GT segments that are not detected
# destPred: destination for wrongly predicted segments
# write_flag: True is files need to be written, else False (only computes TIoU)
def get_wrong_vid_segments(vid_gt_dict, test_list, destGT, destPred, thresh, write_flag):
    vid_key = vid_gt_dict.keys()[0]
    gt_list = vid_gt_dict[vid_key]     # get the gt list of segments

    # calculate the value
    N_gt = len(gt_list)
    M_test = len(test_list)
    if N_gt==0:
        print "No ground truth segments for video."
    if M_test==0:
        print "No predicted segments for video."
        
    if N_gt==0 or M_test==0:
        return 0
    
    # For all gt shots
    tiou_all_gt = 0
    for idx, gt_shot in enumerate(gt_list):
        max_gt_shot = 0
        for test_shot in test_list:
            # if segments are not disjoint, i.e., an overlap exists
            # end of one segment >= start of other segment
            if not (gt_shot[1] < test_shot[0] or test_shot[1] < gt_shot[0]):
                max_gt_shot = max(max_gt_shot, get_iou(gt_shot, test_shot))
        # check if gt segment has <= threshold
        if write_flag and max_gt_shot <= thresh:
            write_segment(vid_key, destGT, idx, gt_shot)
        tiou_all_gt += max_gt_shot
    # For all test shots
    tiou_all_test = 0
    for idx, test_shot in enumerate(test_list):
        max_test_shot = 0
        for gt_shot in gt_list:
            # if segments are not disjoint, i.e., an overlap exists
            if not (gt_shot[1] < test_shot[0] or test_shot[1] < gt_shot[0]):
                max_test_shot = max(max_test_shot, get_iou(gt_shot, test_shot))
        # check if predicted segment has <= threshold
        if write_flag and max_test_shot <= thresh:
            write_segment(vid_key, destPred, idx, test_shot)
        tiou_all_test += max_test_shot
    
    vid_tiou = ((tiou_all_gt/N_gt)+(tiou_all_test/M_test))/2.
    print "TIoU for video : "+str(vid_tiou)
    return vid_tiou
    
# vid_key is the complete subfolder path to the source video.
# destPath: folder where file has to be written.
# idx : Segment No. in the specific video.(for uniquely naming the destination files)
# segment is a tuple [start_frame, end_frame]
def write_segment(vid_key, destPath, idx, segment):    
    cap = cv2.VideoCapture(os.path.join(DATASET_PREFIX, vid_key))    # open video object

    if not cap.isOpened():
        print "Could not open the video file !! Abort !!"
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    (w, h) = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # first split by '/', then by .
    filename = ((vid_key.rsplit('/',1)[1]).rsplit('.',1)[0])+"_"+str(idx)+".avi"
    dest_filename = os.path.join(destPath, filename)
    print "File : "+dest_filename
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dest_filename, fourcc, fps, (w, h))

    # segment is a tuple like (start, end)
    i = segment[0]
    direction = 1   # default move forward
    print "Shot : "+ str(segment)+ " ## Count : "+str(segment[1]-segment[0]+1)
    while cap.isOpened() and i<=segment[1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
#            cv2.imshow("Frame", frame)
            out.write(frame)
#            direction = waitTillEscPressed()
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
#        if direction==3:    # next video
#            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# get the tiou value for s = {s1, s2, ..., sN} and s' = {s'1, s'2, ..., s'M}
def get_vid_tiou(gt_list, test_list):
    # calculate the value
    N_gt = len(gt_list)
    M_test = len(test_list)
    if N_gt==0 or M_test==0:
        return 0
    # For all gt shots
    tiou_all_gt = 0
    for gt_shot in gt_list:
        max_gt_shot = 0
        for test_shot in test_list:
            # if segments are not disjoint, i.e., an overlap exists
            if not (gt_shot[1] < test_shot[0] or test_shot[1] < gt_shot[0]):
                max_gt_shot = max(max_gt_shot, get_iou(gt_shot, test_shot))
        tiou_all_gt += max_gt_shot
    # For all test shots
    tiou_all_test = 0
    for test_shot in test_list:
        max_test_shot = 0
        for gt_shot in gt_list:
            # if segments are not disjoint, i.e., an overlap exists
            if not (gt_shot[1] < test_shot[0] or test_shot[1] < gt_shot[0]):
                max_test_shot = max(max_test_shot, get_iou(gt_shot, test_shot))
        tiou_all_test += max_test_shot
    
    vid_tiou = ((tiou_all_gt/N_gt)+(tiou_all_test/M_test))/2.
    print "TIoU for video : "+str(vid_tiou)
    return vid_tiou

# calculate iou (using frame counts) between two segments
# function is called only when overlap exists
def get_iou(gt_shot, test_shot):
    # if overlap exists
    t = [gt_shot[0], gt_shot[1], test_shot[0], test_shot[1]]
    upper_b = max(t)
    lower_b = min(t)
    union = upper_b - lower_b + 1.0
    t.remove(upper_b)
    t.remove(lower_b)
    intersection = max(t) - min(t) + 1.0  # remaining values
    return (intersection/union)


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
    gt_dir = os.path.join(SUPPORTING_FILES_PATH, GT_SHOTS)
    shotsFile = os.path.join(SUPPORTING_FILES_PATH, "segment_filt/cricShots_hdiffGray_naive_v1_filt60.json")
    destFolder = "/home/hadoop/VisionWorkspace/Cricket/wrong_segments"
    threshold = 0
    write_flag = True       # True if we need to write wrong segments
    # call the function to write all the wrong segments
    write_wrong_segments(gt_dir, shotsFile, destFolder, threshold, write_flag)
    
    