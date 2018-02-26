## Cricket Shot Detection Project Documentation

This document gives the details of the various scripts and the steps needed to reproduce the results.

A section for future optimizations and modifications to improve the performance and results is given.

_____________________________________________________________________________________________________

### Dataset Description: (folder: dataset_25_fps)

folder structure
dataset_25_fps
	-- youtube : 844 videos 
	-- hotstar : 201 videos
	-- ipl2017 : 27 videos
	-- cpl2015 : 33 videos

Total Videos : 1110
Total Size 	:  ~ 273 GB
Extensions : .avi, .mp4
FPS : 25.0 (const. after conversion using ffmpeg)
Dimensions (W, H) : (640, 360)
Codec for converted videos: .h264 (for those videos that have been converted)

Meta_file : (created on the lines of ActivityNet) 
{
	'youtube/v_.....mp4' : {'fps':25, 'partition':'testing', 'dimensions': [360, 640], 'nFrames':2385},
	'hotstar/....avi' : {'fps':25, 'partition':'validation', 'dimensions': [360, 640], 'nFrames':858602},
	...
	...
}


##### Sample Dataset
WT20 2016 highlight videos. 

Total Videos : 26
Extensions : .avi
Total Size : ~ 1 GB
Dimensions and FPS are same as above


_____________________________________________________________________________________________________

### Steps for execution:

1. Collection of dataset (folder: dataset_creation)
Source: Youtube, Hotstar, etc.

* download_videos.py : Script to take the youtube video ids from a text file and create a bash script which downloads the videos using youtube-dl library. 


2. Preprocessing (folder: preprocess)

* convert_ffmpeg_uniform.py : creates a bash script which uses ffmpeg to convert the raw videos into 25FPS and resolution (360, 640) videos. 

* meta_check.py : checks the metadata of the videos and whether the frames in them are being read by VideoCapture object or not. 

* create_partitions.py : takes the dataset path and creates a meta_info.json file that contains the meta informations of the dataset videos. It is in the form of a dictionary, as explained in the previous section. The calculate_partitions function divides the entire dataset into three parts, by assigning videos to their groups i.e., training, testing and validation, in the ratio of 50%, 25%, 25%. The videos are shuffled and the division is done based on total duration of the videos.
The dataset_25_fps_meta_info.json file is created in the supporting_files folder containing the above information.


3. Features (folder: features)

* extract_hist_diffs.py : 

* extract_sqchi_diffs.py :

* extract_hist_diffs_par.py : Parallelized version of extract_hist_diffs.py over the multiple cores of CPU

* extract_sqchi_diffs_par.py : Parallelized version of extract_hist_diffs.py over the multiple cores of CPU


4. Labeling (semi-automatically) the frames of the video (folder : labels)

SBD Labeling (folder: labels):

* Created scripts SBD_create_labels.py and create_shot_labels.py

* SBD_create_labels.py: for semi-automating the labeling task. It iterates over a set of videos (kept in subfolders in a path). For every video, a corresponding JSON file is created in a similar directory structure. The json has following structure only for CUT boundaries.
{"video_subfolder_file_path": [(preFNum, postFNum), (preFNum, postFNum), ....]}
--> User can use the following keys for traversing
ESC: Move Forward without positive labeling (puts a -ve label)
y  : mark a CUT boundary and move forward
b  : move back and delete last +ve/-ve label

* create_shot_labels.py: for semi-automating the labeling of start and end points of cricket strokes (delivery). For every video, a corresponding JSON file is created in a similar directory structure. The json has following structure.
{"video_subfolder_file_path": [(start1, end1), (start2, end2), ....]}
--> User can use the following keys for traversing and labeling
ESC: Push boolean value of isShot (denotes if current frame is part of cricket shot) to stack and move forward to next consecutive frame pair.
s : change the boolean value of isShot to True, if prevFrame and nextFrame represent a transition to a cricket shot.
f : change the boolean value of isShot to False, if prevFrame and nextFrame represent a transition from a cricket shot.
b : pop the last value from the stack and move back in the frame sequence.

The above two scripts are executed for the sample set of cricket videos and also a partial test set (about 3 GB) taken from the full dataset.


5. Boundary Detection : (folder : boundary_detection)


6. Camera Models : (folder : camera_models)


7. Shot Extraction : (folder : shot_extraction)
Using the predicted shots boundaries, first frame camera models pre-trained on HOG features, and 

* predict_boundaries.py: for predicting CUTs on the histogram features, using a saved pretrained model (pretrained on the same features), save the predictions in a json file and calculate the precision, recall and f-score for the predictions on the ground truth labels of test set samples. 


8. Supporting files (folder : supporting_files)
Contains the meta data files, labels etc. for the full dataset and sample dataset

9. Extracted features (folder : extracted_features)
Contains the extracted features of the full dataset. Histogram differences of RGB and grayscale frames, and chi-squared difference features of the consecutive frames of the full dataset. 

_____________________________________________________________________________________________________

### Description of scripts:


_____________________________________________________________________________________________________

### Code optimizations and modifications

1. Parallelize the extract_hist_diff.py and extract_sqchi_diffs.py over (a) GPU (using cv::cuda)
(b) Over multiple processors (using multiprocess) (b. is done)

2. Verification scripts to see that features extracted on local machine and server are the same. (might differ in .h264 encoding or a different codec, Does it depend on the codec?). May consider a type of Randomized algo (improved time complexity). Checked MD5sum of some files.

3. SVM training of cam1 and cam2 frames on HOG features. Optimize the features, SVM params, and consider different models like Random Forests etc. (shot_extraction/svm_model.py). Check for misclassified examples.

4. 


_____________________________________________________________________________________________________


### New Ideas to be tested

1. Tracking with median/Kalman filtering, and YOLO Tracking.

2. Fourier and Wavelet Transform based filtering using 3D filters.
(Description: FT the motion features, such that the frequency spectrum depicts signal of certain type. The similar motion types will have peaks at specific freq points. Average out over all of the training set videos. -> should reduce the (white) noise to zero and we will get the freq. peaks for different actions. Apply Haar filters or Gabor)

3. BoVW model

4. VideoLSTM model

5. Topic Models approach

6. Understanding neural network motion features in deep neural networks / Linear Neural Networks

7. SBD methods for TRECVid dataset (get results)

8. Optical Flow Visualizations results improvement with HOG features

9. Using CNNs and RNNs for SBD (Sarthak)

10. Detectron from FB Research (Mask R-CNN) for Tracking

11. PyTorch 

12. Dynamic Time Warping Algo for time series comparison.

13. Using shape features and tracking those objects.

14. FlowNet implementation for KTH


Partially completed:

1. Bayesian Modeling of SBD problem on sample dataset.

2. KNN classification over hist_diff values for SBD

3. 


_____________________________________________________________________________________________________

Results and sub-results:

Parallelizing feature extraction 

1. Histogram differences for BGR channels.(Batchsize = 50, nJobs = 10)
Full Dataset : 23832.08 secs = 6.62 hrs

2. Chi Squared differences for BGR channels. (Batchsize = 50, nJobs = 10)
Full Dataset : 23979.83 secs = 6.66 hrs


