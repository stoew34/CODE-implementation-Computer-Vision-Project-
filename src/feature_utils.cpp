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

        // Forward matching: desc1 -> desc2
        std::vector<std::vector<cv::DMatch>> knn_matches_forward;
        matcher_->knnMatch(desc1, desc2, knn_matches_forward, 2);

        // Backward matching: desc2 -> desc1 (for cross-check)
        std::vector<std::vector<cv::DMatch>> knn_matches_backward;
        matcher_->knnMatch(desc2, desc1, knn_matches_backward, 2);

        // Apply Lowe's ratio test + cross-check for robustness
        for (size_t i = 0; i < knn_matches_forward.size(); ++i) {
            if (knn_matches_forward[i].size() < 2) continue;

            // Lowe's ratio test
            if (knn_matches_forward[i][0].distance < ratio_threshold * knn_matches_forward[i][1].distance) {
                cv::DMatch forward_match = knn_matches_forward[i][0];

                // Cross-check: verify backward match agrees
                int train_idx = forward_match.trainIdx;
                if (train_idx < static_cast<int>(knn_matches_backward.size()) &&
                    knn_matches_backward[train_idx].size() > 0) {

                    cv::DMatch backward_match = knn_matches_backward[train_idx][0];

                    // If backward match points back to the same query point, it's consistent
                    if (backward_match.trainIdx == forward_match.queryIdx) {
                        good_matches.push_back(forward_match);
                    }
                }
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

    NormalizationParams FeatureUtils::normalizeBilateralPoints(std::vector<BilateralPoint8D>& points) {
        NormalizationParams params;

        if (points.empty()) return params;

        // Compute mean for spatial coordinates
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

        // Store parameters
        params.mean_x = mean_x;
        params.mean_y = mean_y;
        params.scale = scale;

        // Apply normalization
        applyNormalization(points, params);

        return params;
    }

    void FeatureUtils::applyNormalization(std::vector<BilateralPoint8D>& points,
                                           const NormalizationParams& params) {
        const double pi = 3.14159265358979323846;

        for (auto& p : points) {
            // Normalize spatial coordinates
            p.x = (p.x - params.mean_x) * params.scale;
            p.y = (p.y - params.mean_y) * params.scale;

            // Scale motion vectors
            p.u *= params.scale;
            p.v *= params.scale;

            // CRITICAL: xp and yp must maintain relationship xp=x+u, yp=y+v after normalization
            p.xp = p.x + p.u;
            p.yp = p.y + p.v;

            // Normalize orientations from degrees to [-1, 1]
            // OpenCV keypoint angles are in degrees [0, 360)
            p.o1 = (p.o1 * pi / 180.0) / pi;  // Convert to [-1, 1]
            p.o2 = (p.o2 * pi / 180.0) / pi;
        }
    }

} // namespace code