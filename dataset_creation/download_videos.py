# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:25:25 2017

@author: Arpan

Description: Download youtube videos using youtube IDs. Code taken and modified 
from Crawler of ActivityNet.
"""

from argparse import ArgumentParser
import glob
import os

# Refer the code from 
# https://github.com/arpane4c5/ActivityNet/blob/master/Crawler/run_crosscheck.py
def crosscheck_videos(video_path, ids_file):
    # Get existing videos
    existing_vids = glob.glob("%s/*.mp4" % video_path)
    for idx, vid in enumerate(existing_vids):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) == 13:
            existing_vids[idx] = basename[2:]
        elif len(basename) == 11:
            existing_vids[idx] = basename
        else:
            raise RuntimeError("Unknown filename format: %s", vid)
    # Read an get video IDs from annotation file
    print video_path
    print ids_file
    with open(ids_file) as f:
        all_vids = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    all_vids = [x.strip() for x in all_vids]
    new_ids = []
    for id in all_vids:
    	if len(id) == 28:
    		if id[-11:] not in new_ids:
        		new_ids.append(id[-11:])
        else:
        	if id not in new_ids:
        		new_ids.append(id)
    print new_ids
    non_existing_videos = []
    # for vid in all_vids:
    #     if vid in existing_vids:
    #         continue
    #     else:
    #         non_existing_videos.append(vid)
    for vid in new_ids:
    	if vid in existing_vids:
    		continue
    	else:
    		non_existing_videos.append(vid)

    return non_existing_videos

def main(video_path, ids_file, output_filename):
    # find the videos that do not exist in the local directory
    non_existing_videos = crosscheck_videos(video_path, ids_file)
    print non_existing_videos    
    print "No of non-existing videos = {}" .format(len(non_existing_videos))
    filename = os.path.join(video_path, "v_%s.mp4")
    cmd_base = "youtube-dl -f best -f mp4 "
    cmd_base += '"https://www.youtube.com/watch?v=%s" '
    cmd_base += '-o "%s"' % filename
    # create commands to download youtube videos and write them in output file
    with open(output_filename, "w") as fobj:
        for vid in non_existing_videos:
            cmd = cmd_base % (vid, vid)
            fobj.write("%s\n" % cmd)

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to double check video content.")
    parser.add_argument("video_path", help="Where are located the videos? (Full path)")
    parser.add_argument("ids_file", help="Where is the annotation file?")
    parser.add_argument("output_filename", help="Output script location.")
    args = vars(parser.parse_args())
    main(**args)
    
    # get lists of ids 
#    a = [1,2,3,4,5]
#    b = [3,4,5,6,7]
#    a.extend(b)
#    res = list(set(a))  # remove duplicate entries
    
    #idFile = "D:\\Workspaces\\PythonOpenCV\\ActivityProjPy\\youtube_ids.txt"
    
