#include "feature_matcher.h"
#include <algorithm>

namespace code {

    FeatureMatcher::FeatureMatcher(const Config& config) : config_(config) {
        // Use FLANN for floating point descriptors (SIFT)
        matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    std::vector<FeatureMatch> FeatureMatcher::matchWithThreshold(
        const std::vector<cv::KeyPoint>& kpts1,
        const std::vector<cv::KeyPoint>& kpts2,
        const cv::Mat& desc1,
        const cv::Mat& desc2,
        float threshold) const {

        std::vector<FeatureMatch> matches;

        if (desc1.empty() || desc2.empty() || kpts1.empty() || kpts2.empty()) {
            return matches;
        }

        // Perform KNN matching
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_->knnMatch(desc1, desc2, knn_matches, 2);

        // Apply ratio test
        std::vector<cv::DMatch> good_matches = ratioTest(knn_matches, threshold);

        // Convert to FeatureMatch objects
        for (const auto& dmatch : good_matches) {
            if (dmatch.queryIdx < static_cast<int>(kpts1.size()) &&
                dmatch.trainIdx < static_cast<int>(kpts2.size())) {
                matches.emplace_back(kpts1[dmatch.queryIdx],
                    kpts2[dmatch.trainIdx],
                    dmatch);
            }
        }

        return matches;
    }

    std::vector<BilateralPoint8D> FeatureMatcher::matchesToBilateralPoints(
        const std::vector<FeatureMatch>& matches) const {

        std::vector<BilateralPoint8D> points;
        points.reserve(matches.size());

        for (const auto& match : matches) {
            points.push_back(match.bilateral_point);
        }

        return points;
    }

    std::vector<FeatureMatch> FeatureMatcher::applyMutualConsistency(
        const std::vector<FeatureMatch>& matches,
        const cv::Mat& desc1,
        const cv::Mat& desc2) const {

        if (matches.empty()) return {};

        // Perform matching in both directions
        std::vector<std::vector<cv::DMatch>> knn_forward, knn_backward;
        matcher_->knnMatch(desc1, desc2, knn_forward, 1);
        matcher_->knnMatch(desc2, desc1, knn_backward, 1);

        // Convert to simple DMatch vectors
        std::vector<cv::DMatch> forward_matches, backward_matches;
        for (const auto& match_list : knn_forward) {
            if (!match_list.empty()) {
                forward_matches.push_back(match_list[0]);
            }
        }
        for (const auto& match_list : knn_backward) {
            if (!match_list.empty()) {
                backward_matches.push_back(match_list[0]);
            }
        }

        // Apply cross-check
        std::vector<cv::DMatch> consistent_matches = crossCheck(forward_matches, backward_matches);

        // Convert back to FeatureMatch
        std::vector<FeatureMatch> result;
        // Implementation would convert consistent_matches to FeatureMatch
        // This is simplified

        return result;
    }

    std::vector<cv::DMatch> FeatureMatcher::ratioTest(
        const std::vector<std::vector<cv::DMatch>>& knn_matches,
        float ratio) const {

        std::vector<cv::DMatch> good_matches;

        for (size_t i = 0; i < knn_matches.size(); ++i) {
            if (knn_matches[i].size() < 2) continue;

            const cv::DMatch& m1 = knn_matches[i][0];
            const cv::DMatch& m2 = knn_matches[i][1];

            if (m1.distance < ratio * m2.distance) {
                good_matches.push_back(m1);
            }
        }

        return good_matches;
    }

    std::vector<cv::DMatch> FeatureMatcher::crossCheck(
        const std::vector<cv::DMatch>& matches_forward,
        const std::vector<cv::DMatch>& matches_backward) const {

        std::vector<cv::DMatch> good_matches;

        // Create a map for backward matches
        std::vector<int> backward_match_idx(matches_backward.size(), -1);
        for (size_t i = 0; i < matches_backward.size(); ++i) {
            if (matches_backward[i].trainIdx < static_cast<int>(backward_match_idx.size())) {
                backward_match_idx[matches_backward[i].trainIdx] = i;
            }
        }

        // Check mutual consistency
        for (const auto& forward_match : matches_forward) {
            int query_idx = forward_match.queryIdx;
            int train_idx = forward_match.trainIdx;

            if (train_idx < static_cast<int>(backward_match_idx.size()) &&
                backward_match_idx[train_idx] >= 0) {

                const cv::DMatch& backward_match = matches_backward[backward_match_idx[train_idx]];

                if (backward_match.trainIdx == query_idx) {
                    // Mutual best match
                    good_matches.push_back(forward_match);
                }
            }
        }

        return good_matches;
    }

} // namespace code