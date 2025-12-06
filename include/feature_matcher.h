#pragma once

#include "types.h"
#include "config.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace code {

    class FeatureMatcher {
    public:
        explicit FeatureMatcher(const Config& config);

        // Match features with different thresholds (Fig 6)
        std::vector<FeatureMatch> matchWithThreshold(
            const std::vector<cv::KeyPoint>& kpts1,
            const std::vector<cv::KeyPoint>& kpts2,
            const cv::Mat& desc1,
            const cv::Mat& desc2,
            float threshold) const;

        // Convert matches to bilateral points
        std::vector<BilateralPoint8D> matchesToBilateralPoints(
            const std::vector<FeatureMatch>& matches) const;

        // Apply mutual consistency check
        std::vector<FeatureMatch> applyMutualConsistency(
            const std::vector<FeatureMatch>& matches,
            const cv::Mat& desc1,
            const cv::Mat& desc2) const;

    private:
        Config config_;
        cv::Ptr<cv::DescriptorMatcher> matcher_;

        // Lowe's ratio test
        std::vector<cv::DMatch> ratioTest(
            const std::vector<std::vector<cv::DMatch>>& knn_matches,
            float ratio) const;

        // Cross-check matching
        std::vector<cv::DMatch> crossCheck(
            const std::vector<cv::DMatch>& matches_forward,
            const std::vector<cv::DMatch>& matches_backward) const;
    };

} // namespace code