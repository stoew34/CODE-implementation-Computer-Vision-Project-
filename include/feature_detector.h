#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include "types.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

namespace code {

class FeatureDetector {
public:
    explicit FeatureDetector(const CodeParameters& params);
    
    // Simulate A-SIFT by detecting SIFT features under multiple affine transformations
    void detectAndCompute(const cv::Mat& image,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors);
    
private:
    CodeParameters params_;
    cv::Ptr<cv::SIFT> sift_detector_;
    
    // Generate affine transformation matrices
    std::vector<cv::Mat> generateAffineTransforms();
    
    // Apply affine transformation and detect features
    void detectUnderTransform(const cv::Mat& image,
                            const cv::Mat& transform,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::Mat& descriptors);
};

}

#endif // FEATURE_DETECTOR_H
