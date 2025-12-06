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
        auto bilateral_points = extractBilateralPoints(selected_matches);

        // CRITICAL: Store normalization parameters for reuse everywhere
        NormalizationParams norm_params = feature_utils_->normalizeBilateralPoints(bilateral_points);

        auto likelihood_result = step3_likelihoodRegression(bilateral_points, centroids);

        if (!likelihood_result.converged) {
            logWarning("Likelihood regression did not converge");
        }

        // Apply likelihood filter (points already normalized above)
        auto likelihood_inliers = likelihood_regression_->filterByLikelihood(
            bilateral_points, likelihood_result, centroids);

        std::vector<FeatureMatch> filtered_matches;
        for (size_t i = 0; i < selected_matches.size(); ++i) {
            if (likelihood_inliers[i]) {
                filtered_matches.push_back(selected_matches[i]);
            }
        }

        logInfo("Matches after likelihood filter: " + std::to_string(filtered_matches.size()));

        // Early return if no matches pass likelihood filter
        if (filtered_matches.empty()) {
            logWarning("No matches passed likelihood filter");
            return {};
        }

        // Step 4: Bilaterally varying affine regression
        auto affine_results = step4_affineRegression(filtered_matches, centroids, norm_params);

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
            likelihood_result, affine_results, centroids, norm_params);

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
        const std::vector<BilateralPoint8D>& normalized_points,
        std::vector<BilateralPoint8D>& centroids) {

        // Points are already normalized - use them directly
        // Cluster points for accelerated regression (Sec 2.2)
        centroids = clusterer_->cluster(normalized_points);
        logInfo("Clustered into " + std::to_string(centroids.size()) + " centroids");

        // Compute likelihood function (Eqn 21)
        return likelihood_regression_->computeLikelihoodFunction(normalized_points);
    }

    std::vector<RegressionResult> CodePipeline::step4_affineRegression(
        const std::vector<FeatureMatch>& filtered_matches,
        const std::vector<BilateralPoint8D>& centroids,
        const NormalizationParams& norm_params) {

        auto bilateral_points = extractBilateralPoints(filtered_matches);

        // Use SAME normalization as step3
        feature_utils_->applyNormalization(bilateral_points, norm_params);

        // Compute bilaterally varying affine models (Eqns 24-27)
        return affine_regression_->computeAffineModels(bilateral_points, centroids);
    }

    std::vector<FeatureMatch> CodePipeline::step5_finalFiltering(
        const std::vector<cv::KeyPoint>& kpts1,
        const std::vector<cv::KeyPoint>& kpts2,
        const cv::Mat& desc1, const cv::Mat& desc2,
        const RegressionResult& likelihood_result,
        const std::vector<RegressionResult>& affine_results,
        const std::vector<BilateralPoint8D>& centroids,
        const NormalizationParams& norm_params) {

        auto all_matches = feature_utils_->matchDescriptors(desc1, desc2,
            config_.final_threshold);
        auto feature_matches = feature_utils_->convertMatches(kpts1, kpts2, all_matches);

        logInfo("Step5: Re-matched features: " + std::to_string(feature_matches.size()));

        if (feature_matches.empty()) {
            return {};
        }

        auto bilateral_points = extractBilateralPoints(feature_matches);

        // CRITICAL: Use the SAME normalization parameters from training
        feature_utils_->applyNormalization(bilateral_points, norm_params);

        // Apply likelihood filter
        auto likelihood_inliers = likelihood_regression_->filterByLikelihood(
            bilateral_points, likelihood_result, centroids);

        int likelihood_count = std::count(likelihood_inliers.begin(), likelihood_inliers.end(), true);
        logInfo("Step5: After likelihood filter: " + std::to_string(likelihood_count));
        logInfo("Step5: bilateral_points size: " + std::to_string(bilateral_points.size()));
        logInfo("Step5: centroids size: " + std::to_string(centroids.size()));
        logInfo("Step5: affine_results size: " + std::to_string(affine_results.size()));

        // Apply spatial consistency filter
        auto spatial_inliers = affine_regression_->filterBySpatialConsistency(
            bilateral_points, affine_results, centroids);

        int spatial_count = std::count(spatial_inliers.begin(), spatial_inliers.end(), true);
        logInfo("Step5: After spatial filter: " + std::to_string(spatial_count));

        // Combine filters
        std::vector<FeatureMatch> final_matches;
        for (size_t i = 0; i < feature_matches.size(); ++i) {
            if (likelihood_inliers[i] && spatial_inliers[i]) {
                final_matches.push_back(feature_matches[i]);
            }
        }

        logInfo("Step5: After combining filters: " + std::to_string(final_matches.size()));

        // Apply geometric verification (RANSAC with fundamental matrix)
        final_matches = applyGeometricVerification(final_matches);
        logInfo("Step5: After geometric verification: " + std::to_string(final_matches.size()));

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

    std::vector<FeatureMatch> CodePipeline::applyGeometricVerification(
        const std::vector<FeatureMatch>& matches) {

        if (matches.size() < 8) {  // Need at least 8 points for fundamental matrix
            return matches;
        }

        // Extract point correspondences
        std::vector<cv::Point2f> pts1, pts2;
        pts1.reserve(matches.size());
        pts2.reserve(matches.size());

        for (const auto& match : matches) {
            pts1.push_back(cv::Point2f(match.kp1.pt.x, match.kp1.pt.y));
            pts2.push_back(cv::Point2f(match.kp2.pt.x, match.kp2.pt.y));
        }

        // Use RANSAC to find fundamental matrix and filter outliers
        // Balanced threshold for quality matches
        std::vector<uchar> inlier_mask;
        cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 1.5, 0.999, inlier_mask);

        if (F.empty() || inlier_mask.empty()) {
            return matches;  // Return original if RANSAC fails
        }

        // Filter matches based on inlier mask + additional epipolar constraint verification
        std::vector<FeatureMatch> verified_matches;
        verified_matches.reserve(matches.size());

        for (size_t i = 0; i < matches.size(); ++i) {
            if (inlier_mask[i]) {
                // Additional verification: compute epipolar error explicitly
                cv::Mat pt1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
                cv::Mat pt2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);

                // Epipolar constraint: pt2^T * F * pt1 = 0
                cv::Mat error = pt2.t() * F * pt1;
                double epipolar_error = std::abs(error.at<double>(0, 0));

                // Stricter epipolar error for precision (< 1.5 pixels)
                if (epipolar_error < 1.5) {
                    // Additional check: verify scale and orientation consistency
                    const auto& m = matches[i];
                    float scale_ratio = m.kp2.size / (m.kp1.size + 1e-6f);
                    float angle_diff = std::abs(m.kp2.angle - m.kp1.angle);
                    if (angle_diff > 180.0f) angle_diff = 360.0f - angle_diff;

                    // Balanced constraints: moderate scale (0.4x to 2.5x) and rotation (< 100 degrees)
                    bool reasonable_scale = (scale_ratio > 0.4f && scale_ratio < 2.5f);
                    bool reasonable_angle = (angle_diff < 100.0f);

                    if (reasonable_scale && reasonable_angle) {
                        verified_matches.push_back(matches[i]);
                    }
                }
            }
        }

        int num_outliers = matches.size() - verified_matches.size();
        if (num_outliers > 0) {
            logInfo("Geometric verification removed " + std::to_string(num_outliers) +
                    " outliers (" + std::to_string(verified_matches.size()) + "/" +
                    std::to_string(matches.size()) + " kept)");
        }

        return verified_matches;
    }

} // namespace code