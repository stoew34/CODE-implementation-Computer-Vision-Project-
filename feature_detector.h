#pragma once

#include "types.h"
#include "config.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace code {

    class FeatureDetector {
    public:
        explicit FeatureDetector(const Config& config);

        // Detect A-SIFT like features
        std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat& image) const;

        // Compute descriptors
        cv::Mat computeDescriptors(const cv::Mat& image,
            const std::vector<cv::KeyPoint>& keypoints) const;

        // Full detection
        void detectAndCompute(const cv::Mat& image,
            std::vector<cv::KeyPoint>& keypoints,
            cv::Mat& descriptors) const;

        // Apply Hartley normalization (mentioned in Sec 3.2)
        void normalizeCoordinates(std::vector<cv::Point2f>& points,
            cv::Mat& transform) const;

    private:
        Config config_;

        // Simulate affine transformations (A-SIFT)
        void applyAffineTransforms(const cv::Mat& image,
            std::vector<cv::Mat>& transformed_images,
            std::vector<cv::Mat>& transforms) const;

        // Merge features from different affine views
        void mergeFeatures(const std::vector<std::vector<cv::KeyPoint>>& all_keypoints,
            const std::vector<cv::Mat>& all_descriptors,
            std::vector<cv::KeyPoint>& merged_keypoints,
            cv::Mat& merged_descriptors) const;
    };

} // namespace code