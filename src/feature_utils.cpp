#include "feature_utils.h"
#include <numeric>
#include <algorithm>

namespace code {

    FeatureUtils::FeatureUtils(const Config& config) : config_(config) {
        // Create SIFT with optimized parameters for better quality
        detector_ = cv::SIFT::create(
            config_.max_features * 2,  // Detect more, filter later
            4,  // More octave layers for better scale detection
            config_.contrast_threshold,
            config_.edge_threshold,
            config_.sigma);

        matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    void FeatureUtils::detectAndCompute(const cv::Mat& image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors) {
        detector_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        // Sort by response (feature strength) and keep the best
        if (keypoints.size() > static_cast<size_t>(config_.max_features)) {
            // Create indices sorted by response (descending)
            std::vector<size_t> indices(keypoints.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                [&keypoints](size_t a, size_t b) {
                    return keypoints[a].response > keypoints[b].response;
                });

            // Keep only the strongest features
            std::vector<cv::KeyPoint> filtered_kpts;
            cv::Mat filtered_desc;
            filtered_kpts.reserve(config_.max_features);

            for (size_t i = 0; i < static_cast<size_t>(config_.max_features); ++i) {
                filtered_kpts.push_back(keypoints[indices[i]]);
                filtered_desc.push_back(descriptors.row(indices[i]));
            }

            keypoints = std::move(filtered_kpts);
            descriptors = filtered_desc;
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

            cv::DMatch forward_best = knn_matches_forward[i][0];
            cv::DMatch forward_second = knn_matches_forward[i][1];

            // Lowe's ratio test with adaptive threshold
            // For very distinctive features, be more strict
            double adaptive_ratio = ratio_threshold;
            if (forward_best.distance < 50.0) {  // Very good match
                adaptive_ratio = std::min(ratio_threshold, 0.75f);
            }

            if (forward_best.distance < adaptive_ratio * forward_second.distance) {
                // Cross-check: verify backward match agrees
                int train_idx = forward_best.trainIdx;
                if (train_idx < static_cast<int>(knn_matches_backward.size()) &&
                    knn_matches_backward[train_idx].size() > 0) {

                    cv::DMatch backward_match = knn_matches_backward[train_idx][0];

                    // If backward match points back to the same query point, it's consistent
                    if (backward_match.trainIdx == forward_best.queryIdx) {
                        // Additional check: backward ratio test
                        if (knn_matches_backward[train_idx].size() >= 2) {
                            cv::DMatch backward_second = knn_matches_backward[train_idx][1];
                            if (backward_match.distance < ratio_threshold * backward_second.distance) {
                                good_matches.push_back(forward_best);
                            }
                        } else {
                            good_matches.push_back(forward_best);
                        }
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