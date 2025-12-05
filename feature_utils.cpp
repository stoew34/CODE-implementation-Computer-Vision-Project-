#include "feature_utils.h"

namespace code {

    FeatureUtils::FeatureUtils(const Config& config) : config_(config) {
        detector_ = cv::SIFT::create(config_.max_features,
            3,  // nOctaveLayers
            config_.contrast_threshold,
            config_.edge_threshold,
            config_.sigma);

        matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    void FeatureUtils::detectAndCompute(const cv::Mat& image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors) {
        detector_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        // Limit features if necessary
        if (keypoints.size() > static_cast<size_t>(config_.max_features)) {
            keypoints.resize(config_.max_features);
            descriptors = descriptors.rowRange(0, config_.max_features);
        }
    }

    std::vector<cv::DMatch> FeatureUtils::matchDescriptors(
        const cv::Mat& desc1,
        const cv::Mat& desc2,
        float ratio_threshold) {

        std::vector<cv::DMatch> good_matches;

        if (desc1.empty() || desc2.empty()) {
            return good_matches;
        }

        // KNN matching
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(desc1, desc2, knn_matches, 2);

        // Lowe's ratio test
        for (size_t i = 0; i < knn_matches.size(); ++i) {
            if (knn_matches[i].size() < 2) continue;

            if (knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        return good_matches;
    }

    std::vector<FeatureMatch> FeatureUtils::convertMatches(
        const std::vector<cv::KeyPoint>& kpts1,
        const std::vector<cv::KeyPoint>& kpts2,
        const std::vector<cv::DMatch>& matches) {

        std::vector<FeatureMatch> feature_matches;

        for (const auto& dmatch : matches) {
            if (dmatch.queryIdx < static_cast<int>(kpts1.size()) &&
                dmatch.trainIdx < static_cast<int>(kpts2.size())) {
                feature_matches.emplace_back(kpts1[dmatch.queryIdx],
                    kpts2[dmatch.trainIdx],
                    dmatch);
            }
        }

        return feature_matches;
    }

    void FeatureUtils::normalizeBilateralPoints(std::vector<BilateralPoint8D>& points) {
        if (points.empty()) return;

        // Compute mean
        double mean_x = 0.0, mean_y = 0.0;
        for (const auto& p : points) {
            mean_x += p.x;
            mean_y += p.y;
        }
        mean_x /= points.size();
        mean_y /= points.size();

        // Compute scale (average distance from mean)
        double avg_dist = 0.0;
        for (const auto& p : points) {
            double dx = p.x - mean_x;
            double dy = p.y - mean_y;
            avg_dist += std::sqrt(dx * dx + dy * dy);
        }
        avg_dist /= points.size();

        if (avg_dist < 1e-6) avg_dist = 1.0;

        // Normalize to have average distance sqrt(2)
        double scale = std::sqrt(2.0) / avg_dist;

        for (auto& p : points) {
            p.x = (p.x - mean_x) * scale;
            p.y = (p.y - mean_y) * scale;
            p.xp = (p.xp - mean_x) * scale;
            p.yp = (p.yp - mean_y) * scale;
            // u and v are differences, so they get scaled too
            p.u *= scale;
            p.v *= scale;
        }
    }

} // namespace code