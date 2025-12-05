#include "feature_detector.h"
#include <opencv2/xfeatures2d.hpp>
#include <cmath>

namespace code {

    FeatureDetector::FeatureDetector(const Config& config) : config_(config) {}

    std::vector<cv::KeyPoint> FeatureDetector::detectKeypoints(const cv::Mat& image) const {
        // Use SIFT detector (simplified version of A-SIFT)
        auto detector = cv::SIFT::create(config_.max_features,
            3,  // nOctaveLayers
            config_.contrast_threshold,
            config_.edge_threshold,
            config_.sigma);

        std::vector<cv::KeyPoint> keypoints;
        detector->detect(image, keypoints);

        return keypoints;
    }

    cv::Mat FeatureDetector::computeDescriptors(const cv::Mat& image,
        const std::vector<cv::KeyPoint>& keypoints) const {
        auto detector = cv::SIFT::create(config_.max_features,
            3,
            config_.contrast_threshold,
            config_.edge_threshold,
            config_.sigma);

        cv::Mat descriptors;
        detector->compute(image, keypoints, descriptors);

        return descriptors;
    }

    void FeatureDetector::detectAndCompute(const cv::Mat& image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors) const {
        // Simplified version - in full A-SIFT we would apply affine transforms
        auto detector = cv::SIFT::create(config_.max_features,
            3,
            config_.contrast_threshold,
            config_.edge_threshold,
            config_.sigma);

        detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        // Limit number of features if necessary
        if (keypoints.size() > static_cast<size_t>(config_.max_features)) {
            keypoints.resize(config_.max_features);
            descriptors = descriptors.rowRange(0, config_.max_features);
        }
    }

    void FeatureDetector::normalizeCoordinates(std::vector<cv::Point2f>& points,
        cv::Mat& transform) const {
        if (points.empty()) return;

        // Compute mean
        cv::Point2f mean(0, 0);
        for (const auto& p : points) {
            mean += p;
        }
        mean.x /= points.size();
        mean.y /= points.size();

        // Compute average distance from mean
        double avg_dist = 0.0;
        for (const auto& p : points) {
            double dx = p.x - mean.x;
            double dy = p.y - mean.y;
            avg_dist += sqrt(dx * dx + dy * dy);
        }
        avg_dist /= points.size();

        if (avg_dist < 1e-6) avg_dist = 1.0;

        // Create normalization transform
        double scale = sqrt(2.0) / avg_dist;
        transform = (cv::Mat_<double>(3, 3) << scale, 0, -mean.x * scale,
            0, scale, -mean.y * scale,
            0, 0, 1);

        // Apply normalization
        for (auto& p : points) {
            p.x = (p.x - mean.x) * scale;
            p.y = (p.y - mean.y) * scale;
        }
    }

    void FeatureDetector::applyAffineTransforms(const cv::Mat& image,
        std::vector<cv::Mat>& transformed_images,
        std::vector<cv::Mat>& transforms) const {
        // Simplified affine simulation
        // In full A-SIFT, multiple affine transformations are applied

        transformed_images.clear();
        transforms.clear();

        // Original image
        transformed_images.push_back(image.clone());
        transforms.push_back(cv::Mat::eye(3, 3, CV_64F));

        // Example affine transforms (simplified)
        std::vector<double> tilts = { 1.0, sqrt(2.0), 2.0, 2.0 * sqrt(2.0) };
        std::vector<double> rotations = { 0.0, M_PI / 6, M_PI / 4, M_PI / 3 };

        for (double tilt : tilts) {
            for (double rotation : rotations) {
                // Create affine transform matrix
                cv::Mat transform = cv::Mat::eye(3, 3, CV_64F);

                // Apply tilt (simplified)
                transform.at<double>(0, 0) = 1.0 / tilt;

                // Apply rotation
                cv::Mat rotation_mat = cv::getRotationMatrix2D(
                    cv::Point2f(image.cols / 2.0, image.rows / 2.0),
                    rotation * 180.0 / M_PI, 1.0);

                cv::Mat warped;
                cv::warpAffine(image, warped, rotation_mat, image.size());

                transformed_images.push_back(warped);

                // Combine transforms
                cv::Mat full_transform = cv::Mat::eye(3, 3, CV_64F);
                rotation_mat.copyTo(full_transform(cv::Rect(0, 0, 3, 2)));
                transforms.push_back(full_transform);
            }
        }
    }

    void FeatureDetector::mergeFeatures(
        const std::vector<std::vector<cv::KeyPoint>>& all_keypoints,
        const std::vector<cv::Mat>& all_descriptors,
        std::vector<cv::KeyPoint>& merged_keypoints,
        cv::Mat& merged_descriptors) const {

        merged_keypoints.clear();

        // Simple merging: take all features
        for (const auto& keypoints : all_keypoints) {
            merged_keypoints.insert(merged_keypoints.end(),
                keypoints.begin(), keypoints.end());
        }

        // Concatenate descriptors
        std::vector<cv::Mat> descriptor_list;
        for (const auto& desc : all_descriptors) {
            if (!desc.empty()) {
                descriptor_list.push_back(desc);
            }
        }

        if (!descriptor_list.empty()) {
            cv::vconcat(descriptor_list, merged_descriptors);
        }
    }

} // namespace code