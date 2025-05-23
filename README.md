# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.

2. Download dat/yolo/yolov3.weights with Git LFS

   or `! wget "https://pjreddie.com/media/files/yolov3.weights"`

3. Make a build directory in the top level project directory: `mkdir build && cd build`

4. Compile: `cmake .. && make`

5. Run it: `./3D_object_tracking`.

   

## Steps

### FP.1 Match 3D Objects

Method - "matchBoundingBoxes": Input parameters - previous & current data frames, Output - ids of matching regions of interest (matches with highest number of keypoint correspondences are returned). 


### FP.2 Compute lidar-based TTC

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements  based on median value.

### FP.3 Associate Keypoint Correspondences with Bounding Boxes

Associating keypoint correspondences to the bounding boxes which enclose them - based on mean distance of the selected points. 

### FP.4 Compute Camera-based TTC

Compute the time-to-collision in second for all matched 3D objects using camera measurements  based on median value.
 

## FP.5 Performance Evaluation 1

No examples of  TTC estimate of the Lidar sensor that seem not plausible (probably beacuse median values were used to remove outliers)

 
### FP.6 Performance Evaluation 2

Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

Detector Descriptor TTC Lidar TTC Camera TTC Diff 

SHITOMASI BRISK 11.729290 13.504473 1.775183
--------------------------------------------------------
SHITOMASI BRIEF 11.729290 12.108298 0.379008
--------------------------------------------------------
SHITOMASI ORB 11.729290 11.572136 0.157154
--------------------------------------------------------
SHITOMASI FREAK 11.729290 12.641528 0.912238
--------------------------------------------------------
SHITOMASI SIFT 11.729290 13.266496 1.537206
-------------------------------------------------------- 
FAST BRISK 11.729290 13.848768 2.119478
--------------------------------------------------------
FAST BRIEF 11.729290 17.348754 5.619464
--------------------------------------------------------
FAST ORB 11.729290 12.940983 1.211694
--------------------------------------------------------
FAST FREAK 11.729290 13.895831 2.166541
--------------------------------------------------------
FAST SIFT 11.729290 15.061540 3.332250
--------------------------------------------------------
BRISK BRISK 11.729290 13.960218 2.230928
--------------------------------------------------------
BRISK BRIEF 11.729290 14.820086 3.090796
--------------------------------------------------------
BRISK ORB 11.729290 13.866201 2.136911
--------------------------------------------------------
BRISK FREAK 11.729290 14.268525 2.539235
--------------------------------------------------------
BRISK SIFT 11.729290 14.733666 3.004376
--------------------------------------------------------
ORB BRISK 11.729290 317.035299 305.306010
--------------------------------------------------------
ORB BRIEF 11.729290 28.160930 16.431640
--------------------------------------------------------
ORB ORB 11.729290 18.193555 6.464265
--------------------------------------------------------
AKAZE BRISK 11.729290 12.410020 0.680730
--------------------------------------------------------
AKAZE BRIEF 11.729290 12.500827 0.771537
--------------------------------------------------------
AKAZE ORB 11.729290 12.431672 0.702382
--------------------------------------------------------
AKAZE FREAK 11.729290 12.225710 0.496420
--------------------------------------------------------
AKAZE AKAZE 11.729290 12.436410 0.707120
--------------------------------------------------------
AKAZE SIFT 11.729290 14.878567 3.149277
--------------------------------------------------------
SIFT BRISK 11.729290 12.101918 0.372628
--------------------------------------------------------
SIFT BRIEF 11.729290 11.920189 0.190899
--------------------------------------------------------
SIFT FREAK 11.729290 11.752076 0.022786
--------------------------------------------------------
SIFT SIFT 11.729290 11.882527 0.153237
--------------------------------------------------------


The TOP3 detector / descriptor combinations：
SIFT/FREAK 
SIFT/SIFT 
SIFT/BRIEF 