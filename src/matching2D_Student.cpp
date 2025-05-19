#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k = 2; // number of nearest neighbors to search for
        vector<vector<cv::DMatch>> knn_m;
        matcher->knnMatch(descSource, descRef, knn_m, k); // Finds the k best matches for each descriptor in desc1
        // implement the descriptor distance ratio test with t=0.8       
        for (size_t i=0; i < knn_m.size(); i++) {
            if (knn_m[i][0].distance < knn_m[i][1].distance * 0.8) {
                matches.push_back(knn_m[i][0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0) {
        
        extractor = cv::SIFT::create();
    }
    else
    {
        cerr << "Error: Invalid descriptor type provided: " << descriptorType << endl;
        return;
    }

    // perform feature description  
    extractor->compute(img, keypoints, descriptors);
}
   

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 4;
    int apertureSize = 3;
    int minResponse = 100;
    double k = 0.04;

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
  
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    
    double maxOverlap = 0.0;
    for (int j = 0; j < dst_norm.rows; j++) {
        for (int i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);
            // Check for thershold to filter out weak corners 
            if (response > minResponse) {               
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;
                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it!= keypoints.end(); ++it) {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap) {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response) {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                } // if no overlap, add new key point to the list
            }
        }
    }
 
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
 
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
  
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) 
{
    cv::Ptr<cv::FeatureDetector> detector;
    // select appropriate feature detector
    if (detectorType.compare("FAST") == 0) {    
        detector = cv::FastFeatureDetector::create(30, true, cv::FastFeatureDetector::TYPE_7_12);          
    }
    else if (detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create(30, 3, 1.0f);
    }
    else if (detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();      
    }
    else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();       
    }
    else if (detectorType.compare("SIFT") == 0) {
        detector = cv::SIFT::create();   
    }
    else {
        cerr << "Error: Invalid detector type provided: " << detectorType << endl;
        return;
    }
    detector->detect(img, keypoints);
    if (bVis)
    {
      cv::Mat visImage = img.clone();
      cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      string windowName = detectorType + " Results";
      cv::namedWindow(windowName, 2);
      imshow(windowName, visImage);
      cv::waitKey(0);
    }
}