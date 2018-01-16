#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jan 10 21:47:07 2017

@author: Arpan

Execute this file after converting the dataset using convert_ffmpeg.py file
@Description: To search for videos that do not have FPS=25.0 or (H,W) = (360, 640) 
and create bash file with appropriate ffmpeg conversion command. The rest of the videos
are copied to the destination as they are. Use the bash file to convert the videos.

"""

import os
import cv2

# function to traverse the dataset (containing videos in subfolders) and write cmd 
# to convert video, if not of 25 FPS or 360:640 (h,w). If videos are of 25FPS and (h,w)
# then simply copy the file. If the video is not opened using VideoCapture, then notify user
def make_uniform_dataset(srcFolderPath, destFolderPath, output_filename):
    # iterate over the subfolders in srcFolderPath and convert required videos 
    sfp_lst = os.listdir(srcFolderPath)
    
    #ffmpeg -y -i <input.mp4> -vcodec h264 -vf "scale=640:360,fps=25" <output.mp4>  
    cmd_base = 'ffmpeg -i "%s" ' % os.path.join(srcFolderPath, "%s")
    cmd_base += '-vcodec h264 -vf \"scale=640:360,fps=25\" "%s"' % os.path.join(destFolderPath, "%s")
    
    # create commands to convert files using ffmpeg
    traversed_conv = 0
    traversed_copy = 0
    defective = 0
    with open(output_filename, "w") as fobj:
        for sf in sfp_lst:
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
                        convert = to_be_converted(os.path.join(sub_src_path, vid))
                        sub_path = sf+'/'+vid 
                        # save at the destination, if extracted successfully
                        if convert:    
                            cmd = cmd_base % (sub_path, sub_path)
                            print "Done : "+sf+"/"+vid
                            traversed_conv += 1
                        else:
                            if convert is None:
                                defective += 1
                            else:
                                cmd = 'cp "%s" "%s"' % (os.path.join(sub_src_path, vid), os.path.join(sub_dest_path, vid))
                                traversed_copy += 1
                        fobj.write("%s\n" % cmd)
                        # to stop after successful traversal of 2 videos, if stop != 'all'
                        #if stop != 'all' and traversed == stop:
                        #    break
                    
    print "No. of files that will be converted : "+str(traversed_conv)
    print "No. of files that will be copied : "+str(traversed_copy)
    print "No. of Non-iterable files : "+str(defective)
    tot = traversed_conv + traversed_copy + defective
    print "Total files traversed : "+str(tot)
    if tot == 0:
        print "Check the structure of the dataset folders !!"
    
    return

# function to check whether the video needs to be converted or not. It not, then simply 
# copy the file to destination path.
def to_be_converted(srcVideoPath):
    # get the VideoCapture object
    cap = cv2.VideoCapture(srcVideoPath)
    convert = False
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None

    # check whether video needs to be converted
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps!=25 or h!=360 or w!=640 :
        convert = True
    cap.release()
    return convert
        

if __name__ == '__main__':
    
    srcPath = "/home/arpan/DATA_Drive/Cricket/dataset"
    destPath = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps"
    output_file = "exec_fps_25_ffmpeg.sh"
    
    make_uniform_dataset(srcPath, destPath, output_file)
    
