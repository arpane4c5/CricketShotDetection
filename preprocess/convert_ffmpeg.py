#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:47:07 2017

@author: hadoop

@Description: To convert videos from some format to mp4. It creates a script which can
be executed using bash.

"""

import os

def create_ffmpeg_cmd_file(srcPath, destPath, output_filename):
    if not os.path.exists(destPath):
        os.makedirs(destPath)
        print "Path Created !!"
    
    src_files = os.listdir(srcPath)
    
    cmd_base = 'ffmpeg -i "%s" ' % os.path.join(srcPath, "%s")
    cmd_base += '-vcodec copy -acodec copy "%s"' % os.path.join(destPath, "%s")
    
    # create commands to convert files using ffmpeg
    with open(output_filename, "w") as fobj:
        for vid in src_files:
            vid_prefix = vid.split('.')[0]
            extn = vid.split('.')[-1]
            if not extn in ['mp4', 'avi']:
                cmd = cmd_base % (vid, vid_prefix+'.mp4')
                fobj.write("%s\n" % cmd)
    return

if __name__ == '__main__':
    
    srcPath = "/home/arpan/DATA_Drive/Cricket/hotstar_new"
    destPath = "/home/arpan/DATA_Drive/Cricket/dataset/hotstar_converted"
    output_file = "exec_convert_ffmpeg1.sh"
    
    create_ffmpeg_cmd_file(srcPath, destPath, output_file)
    
