#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:56:15 2018

@author: Arpan

@Description: Evaluation script for cricket shots.
Compute the Temporal IoU metric for the labeled cricket shots for the test set sample
Refer: ActivityNet localization evaluations script. Here only single action is defined
therefore, instead of mean tIoU, we take tIoU.
"""

import json
import os

# Server Params
# This path contains 4 subfolders : youtube, hotstar_converted, ipl2017, cpl2015
#DATASET_PREFIX = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps"  
#SUPPORTING_FILES_PATH = "/home/arpan/VisionWorkspace/shot_detection/supporting_files"
# Local Params
DATASET_PREFIX = "/home/hadoop/VisionWorkspace/Cricket/dataset_25_fps"
SUPPORTING_FILES_PATH = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files"
GT_SHOTS = "dataset_25_fps_val_set_labels/corrected_shots"
DATASET_INFO = "dataset_25_fps_meta_info.json"

# Take the predictions json file and iterate over the ground truth kept inside the folder

def calculate_tIoU(gt_dir, shots_dict):
    # Iterate over the gt files and collect labels into a gt dictionary
    sfp_lst = os.listdir(gt_dir)
    tot_tiou = 0
    tot_segments = 0
    #for sf in sfp_lst:
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
                    vid_tiou = get_vid_tiou(gt_list, test_list)
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

if __name__ == '__main__':
    # Take 
    pred_shots_file = "cricShots_hdiffGray_naive_multi_v1.json"
    with open(os.path.join(SUPPORTING_FILES_PATH, pred_shots_file), 'r') as fp:
        shots_dict = json.load(fp)
        
    tiou = calculate_tIoU(os.path.join(SUPPORTING_FILES_PATH, GT_SHOTS), shots_dict)