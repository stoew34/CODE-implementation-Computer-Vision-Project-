#include "feature_detector.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>

using namespace std;

namespace code {

FeatureDetector::FeatureDetector(const CodeParameters& params)
    : params_(params) {
    sift_detector_ = cv::SIFT::create();
}

vector<cv::Mat> FeatureDetector::generateAffineTransforms() {
    vector<cv::Mat> transforms;
    
    // Identity transform
    transforms.push_back(cv::Mat::eye(2, 3, CV_32F));
    
    // Generate tilts and rotations for A-SIFT simulation
    const vector<float> tilts = {1.0f, 1.5f, 2.0f, 3.0f};
    const int num_rotations = 72; // 5-degree increments
    
    for (float tilt : tilts) {
        if (tilt == 1.0f) continue; // Skip identity (already added)
        
        int rot_steps = static_cast<int>(num_rotations / tilt);
        for (int i = 0; i < rot_steps && transforms.size() < params_.num_affine_samples; ++i) {
            float angle = (360.0f / rot_steps) * i;
            float rad = angle * CV_PI / 180.0f;
            
            // Create affine matrix: rotation then tilt
            cv::Mat A = cv::Mat::eye(2, 3, CV_32F);
            A.at<float>(0, 0) = tilt * cos(rad);
            A.at<float>(0, 1) = -tilt * sin(rad);
            A.at<float>(1, 0) = sin(rad);
            A.at<float>(1, 1) = cos(rad);
            
            transforms.push_back(A);
            
            if (transforms.size() >= params_.num_affine_samples) break;
        }
    }
    
    return transforms;
}

void FeatureDetector::detectUnderTransform(const cv::Mat& image,
                                          const cv::Mat& transform,
                                          vector<cv::KeyPoint>& keypoints,
                                          cv::Mat& descriptors) {
    cv::Mat warped;
    cv::warpAffine(image, warped, transform, image.size());
    
    vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    sift_detector_->detectAndCompute(warped, cv::noArray(), kpts, desc);
    
    // Transform keypoints back to original image coordinates
    cv::Mat inv_transform;
    cv::invertAffineTransform(transform, inv_transform);
    
    for (auto& kpt : kpts) {
        cv::Mat pt = (cv::Mat_<float>(3, 1) << kpt.pt.x, kpt.pt.y, 1.0f);
        cv::Mat transformed = inv_transform * pt;
        kpt.pt.x = transformed.at<float>(0);
        kpt.pt.y = transformed.at<float>(1);
    }
    
    keypoints.insert(keypoints.end(), kpts.begin(), kpts.end());
    descriptors.push_back(desc);
}

void FeatureDetector::detectAndCompute(const cv::Mat& image,
                                      vector<cv::KeyPoint>& keypoints,
                                      cv::Mat& descriptors) {
    keypoints.clear();
    descriptors = cv::Mat();
    
    auto transforms = generateAffineTransforms();
    
    for (const auto& transform : transforms) {
        detectUnderTransform(image, transform, keypoints, descriptors);
    }
    
    cout << "Detected " << keypoints.size() << " features under " 
              << transforms.size() << " affine transformations" << endl;
}

}
