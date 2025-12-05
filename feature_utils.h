#pragma once

#include "types.h"
#include "config.h"

namespace code {

    class FeatureUtils {
    public:
        explicit FeatureUtils(const Config& config);

        // Detect and compute features (simplified A-SIFT)
        void detectAndCompute(const cv::Mat& image,
            std::vector<cv::KeyPoint>& keypoints,
            cv::Mat& descriptors);

        // Match features with ratio test
        std::vector<cv::DMatch> matchDescriptors(
            const cv::Mat& desc1,
            const cv::Mat& desc2,
            float ratio_threshold);

        // Convert matches to FeatureMatch objects
        std::vector<FeatureMatch> convertMatches(
            const std::vector<cv::KeyPoint>& kpts1,
            const std::vector<cv::KeyPoint>& kpts2,
            const std::vector<cv::DMatch>& matches);

        // Apply Hartley normalization (mentioned in Sec 3.2)
        void normalizeBilateralPoints(std::vector<BilateralPoint8D>& points);

    private:
        Config config_;
        cv::Ptr<cv::SIFT> detector_;
        cv::Ptr<cv::DescriptorMatcher> matcher_;
    };

} // namespace code