# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 22:15:08 2016

Description: SBD - Read/write cut positions from/to an xml file. 
            Following TRECVID format.
@author: Bloodhound
"""

from lxml import etree
import os
import cv2
import sys

# frame no convention starting from frame 0 to frame (n-1) for 1st shot 
# srcVideoFile is the path to the src video, 
def write_cuts_to_xml(destFolder, srcVideoFile, cuts_list):
    #print(srcVideoFile)
    import re
    srcVideo = re.split(r'/|\\', srcVideoFile)[-1]
    # create destination folder if not created already
    if not os.path.exists(destFolder):
        os.makedirs(destFolder)
    xmlFileName = os.path.join(destFolder, "test_"+srcVideo.split(".")[0]+".xml")
    # define the root tag
    root = etree.Element('refSeg')
    tree = etree.ElementTree(root)
    root.set('src',srcVideoFile)
    root.set('creationMethod','MANUAL')
    # find total no of frames in the video 
    cap = cv2.VideoCapture(srcVideoFile)
    if not cap.isOpened():
        print "Error opening video file !!"
        sys.exit(0)
    totalFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    cap.release()
    root.set('totalFNum', str(totalFrames))
    # create child tag

    for cut in cuts_list:
        trans = etree.Element('trans')
        trans.set('type', 'CUT')
        trans.set('preFNum', str(cut-1))
        trans.set('postFNum', str(cut))
        root.append(trans)
        
    #print(etree.tostring(root, pretty_print=True))
    tree.write(open(xmlFileName, 'w'), pretty_print=True)
    return
    
def get_cuts_list_from_xml(srcXML):
    # copy from SBD_Evaluate
    cuts_list = list()
    root = etree.parse(srcXML).getroot()

    # Iterate over the transition tags    
    for atype in root.findall('trans'):
        if atype.get('type')=='CUT':
            #print(atype.get('type'), atype.get('preFNum'), atype.get('postFNum'))
            cuts_list.append(int(atype.get('postFNum')))
    # sort the list, if it is not sorted and return
    cuts_list = sorted(cuts_list)
    return cuts_list

if __name__=="__main__":
    srcVideoFolder = "D:/LNMIIT PhD/Thesis Project/ICC WT20"
    srcVideoFile = "ICC WT20 - Afghanistan vs South Africa - Match Highlights.avi"
    destFolder = "D:/Workspaces/PythonOpenCV/ActivityProjPy/frames_dest"
    src = os.path.join(srcVideoFolder, srcVideoFile)
    #write_cuts_to_xml(destFolder, src, [98, 219, 376, 568, 679, 729, 848, 1252, 1359, 1711, 1927, 1991, 2124, 2278, 2354, 2400, 2534, 2848, 2957, 3051, 3180, 3247, 3388, 3508, 3706, 3792, 4031, 4054, 4116, 4131, 4245, 4305])
    srcXML = "D:/Workspaces/PythonOpenCV/ActivityProjPy/sample_frames_dest_200000/xml_cuts/test_ICC WT20 - Afghanistan vs South Africa - Match Highlights.xml"
    
    cuts_list = get_cuts_list_from_xml(srcXML)
    print cuts_list