#pragma once

#include "types.h"
#include "config.h"
#include "feature_utils.h"
#include "bilateral_clusterer.h"
#include "coherence_regression.h"
#include "bilateral_affine.h"

namespace code {

    class CodePipeline {
    public:
        explicit CodePipeline(const Config& config = Config());

        // Main matching function (Fig 6 in paper)
        std::vector<FeatureMatch> matchImages(const cv::Mat& image1,
            const cv::Mat& image2);

    private:
        Config config_;

        // Components
        std::unique_ptr<FeatureUtils> feature_utils_;
        std::unique_ptr<BilateralClusterer> clusterer_;
        std::unique_ptr<CoherenceRegression> likelihood_regression_;
        std::unique_ptr<BilateralAffine> affine_regression_;

        // Pipeline steps
        void step1_detectFeatures(const cv::Mat& img1, const cv::Mat& img2,
            std::vector<cv::KeyPoint>& kpts1,
            std::vector<cv::KeyPoint>& kpts2,
            cv::Mat& desc1, cv::Mat& desc2);

        std::vector<FeatureMatch> step2_initialMatching(
            const std::vector<cv::KeyPoint>& kpts1,
            const std::vector<cv::KeyPoint>& kpts2,
            const cv::Mat& desc1, const cv::Mat& desc2);

        RegressionResult step3_likelihoodRegression(
            const std::vector<BilateralPoint8D>& normalized_points,
            std::vector<BilateralPoint8D>& centroids);

        std::vector<RegressionResult> step4_affineRegression(
            const std::vector<FeatureMatch>& filtered_matches,
            const std::vector<BilateralPoint8D>& centroids,
            const NormalizationParams& norm_params);

        std::vector<FeatureMatch> step5_finalFiltering(
            const std::vector<cv::KeyPoint>& kpts1,
            const std::vector<cv::KeyPoint>& kpts2,
            const cv::Mat& desc1, const cv::Mat& desc2,
            const RegressionResult& likelihood_result,
            const std::vector<RegressionResult>& affine_results,
            const std::vector<BilateralPoint8D>& centroids,
            const NormalizationParams& norm_params);

        // Helper functions
        std::vector<BilateralPoint8D> extractBilateralPoints(
            const std::vector<FeatureMatch>& matches);
    };

} // namespace code