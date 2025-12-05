#include "code_pipeline.h"

namespace code {

    CodePipeline::CodePipeline(const Config& config) : config_(config) {
        feature_utils_ = std::make_unique<FeatureUtils>(config_);
        clusterer_ = std::make_unique<BilateralClusterer>(config_);
        likelihood_regression_ = std::make_unique<CoherenceRegression>(config_);
        affine_regression_ = std::make_unique<BilateralAffine>(config_);
    }

    std::vector<FeatureMatch> CodePipeline::matchImages(const cv::Mat& image1,
        const cv::Mat& image2) {
        logInfo("Starting CODE pipeline (Fig 6)...");

        // Step 1: Detect features
        std::vector<cv::KeyPoint> kpts1, kpts2;
        cv::Mat desc1, desc2;
        step1_detectFeatures(image1, image2, kpts1, kpts2, desc1, desc2);

        auto selected_matches = step2_initialMatching(kpts1, kpts2, desc1, desc2);

        if (selected_matches.empty()) {
            logWarning("No initial matches found");
            return {};
        }

        logInfo("Selected matches for regression: " + std::to_string(selected_matches.size()));

        // Step 3: Likelihood regression (Sec 3)
        std::vector<BilateralPoint8D> centroids;
        auto likelihood_result = step3_likelihoodRegression(selected_matches, centroids);

        if (!likelihood_result.converged) {
            logWarning("Likelihood regression did not converge");
        }

        // Apply likelihood filter
        auto bilateral_points = extractBilateralPoints(selected_matches);
        auto likelihood_inliers = likelihood_regression_->filterByLikelihood(
            bilateral_points, likelihood_result, centroids);

        std::vector<FeatureMatch> filtered_matches;
        for (size_t i = 0; i < selected_matches.size(); ++i) {
            if (likelihood_inliers[i]) {
                filtered_matches.push_back(selected_matches[i]);
            }
        }

        logInfo("Matches after likelihood filter: " + std::to_string(filtered_matches.size()));

        // Step 4: Bilaterally varying affine regression
        auto affine_results = step4_affineRegression(filtered_matches, centroids);

        bool all_converged = true;
        for (const auto& result : affine_results) {
            if (!result.converged) {
                all_converged = false;
                break;
            }
        }

        if (!all_converged) {
            logWarning("Some affine parameters did not converge");
        }

        auto final_matches = step5_finalFiltering(kpts1, kpts2, desc1, desc2,
            likelihood_result, affine_results, centroids);

        logInfo("Final coherent matches: " + std::to_string(final_matches.size()));

        return final_matches;
    }

    void CodePipeline::step1_detectFeatures(const cv::Mat& img1, const cv::Mat& img2,
        std::vector<cv::KeyPoint>& kpts1,
        std::vector<cv::KeyPoint>& kpts2,
        cv::Mat& desc1, cv::Mat& desc2) {
        feature_utils_->detectAndCompute(img1, kpts1, desc1);
        feature_utils_->detectAndCompute(img2, kpts2, desc2);

        logInfo("Detected features: " +
            std::to_string(kpts1.size()) + " in image 1, " +
            std::to_string(kpts2.size()) + " in image 2");
    }

    std::vector<FeatureMatch> CodePipeline::step2_initialMatching(
        const std::vector<cv::KeyPoint>& kpts1,
        const std::vector<cv::KeyPoint>& kpts2,
        const cv::Mat& desc1, const cv::Mat& desc2) {

        auto matches = feature_utils_->matchDescriptors(desc1, desc2,
            config_.initial_threshold);
        return feature_utils_->convertMatches(kpts1, kpts2, matches);
    }

    RegressionResult CodePipeline::step3_likelihoodRegression(
        const std::vector<FeatureMatch>& matches,
        std::vector<BilateralPoint8D>& centroids) {

        auto bilateral_points = extractBilateralPoints(matches);
        feature_utils_->normalizeBilateralPoints(bilateral_points);

        // Cluster points for accelerated regression (Sec 2.2)
        centroids = clusterer_->cluster(bilateral_points);
        logInfo("Clustered into " + std::to_string(centroids.size()) + " centroids");

        // Compute likelihood function (Eqn 21)
        return likelihood_regression_->computeLikelihoodFunction(bilateral_points);
    }

    std::vector<RegressionResult> CodePipeline::step4_affineRegression(
        const std::vector<FeatureMatch>& filtered_matches,
        const std::vector<BilateralPoint8D>& centroids) {

        auto bilateral_points = extractBilateralPoints(filtered_matches);
        feature_utils_->normalizeBilateralPoints(bilateral_points);

        // Compute bilaterally varying affine models (Eqns 24-27)
        return affine_regression_->computeAffineModels(bilateral_points, centroids);
    }

    std::vector<FeatureMatch> CodePipeline::step5_finalFiltering(
        const std::vector<cv::KeyPoint>& kpts1,
        const std::vector<cv::KeyPoint>& kpts2,
        const cv::Mat& desc1, const cv::Mat& desc2,
        const RegressionResult& likelihood_result,
        const std::vector<RegressionResult>& affine_results,
        const std::vector<BilateralPoint8D>& centroids) {

        auto all_matches = feature_utils_->matchDescriptors(desc1, desc2,
            config_.final_threshold);
        auto feature_matches = feature_utils_->convertMatches(kpts1, kpts2, all_matches);

        if (feature_matches.empty()) {
            return {};
        }

        auto bilateral_points = extractBilateralPoints(feature_matches);
        feature_utils_->normalizeBilateralPoints(bilateral_points);

        // Apply likelihood filter
        auto likelihood_inliers = likelihood_regression_->filterByLikelihood(
            bilateral_points, likelihood_result, centroids);

        // Apply spatial consistency filter
        auto spatial_inliers = affine_regression_->filterBySpatialConsistency(
            bilateral_points, affine_results, centroids);

        // Combine filters
        std::vector<FeatureMatch> final_matches;
        for (size_t i = 0; i < feature_matches.size(); ++i) {
            if (likelihood_inliers[i] && spatial_inliers[i]) {
                final_matches.push_back(feature_matches[i]);
            }
        }

        return final_matches;
    }

    std::vector<BilateralPoint8D> CodePipeline::extractBilateralPoints(
        const std::vector<FeatureMatch>& matches) {

        std::vector<BilateralPoint8D> points;
        points.reserve(matches.size());

        for (const auto& match : matches) {
            points.push_back(match.bilateral_point);
        }

        return points;
    }

} // namespace code