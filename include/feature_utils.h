#pragma once

#include "types.h"
#include "config.h"

namespace code {

    // Normalization parameters for bilateral points
    struct NormalizationParams {
        double mean_x = 0.0;
        double mean_y = 0.0;
        double scale = 1.0;
    };

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

        // Apply Hartley normalization and return parameters
        NormalizationParams normalizeBilateralPoints(std::vector<BilateralPoint8D>& points);

        // Apply existing normalization parameters to new points
        void applyNormalization(std::vector<BilateralPoint8D>& points,
                                 const NormalizationParams& params);

    private:
        Config config_;
        cv::Ptr<cv::SIFT> detector_;
        cv::Ptr<cv::DescriptorMatcher> matcher_;
    };

} // namespace code