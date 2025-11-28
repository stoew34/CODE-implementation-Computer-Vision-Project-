#include "feature_detector.h"
#include "feature_matcher.h"
#include "bilateral_clusterer.h" 
#include "coherence_regression.h"
#include <opencv2/opencv.hpp>

int main() {
    // Load images
    cv::Mat image1 = cv::imread("image1.jpg");
    cv::Mat image2 = cv::imread("image2.jpg");

    // Set parameters
    code::CodeParameters params;
    params.match_ratio = 0.86f;  // Relaxed threshold
    params.num_clusters = 100;
    params.max_samples = 30000;
    params.likelihood_threshold = 0.6f;
    params.gamma = 1.0f;
    params.lambda = 1.0f;
    params.epsilon = 0.1f;

    code::FeatureDetector detector(params);
    code::FeatureMatcher matcher(params);

    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    std::vector<code::MatchHypothesis> matches;

    detector.detectAndCompute(image1, kpts1, desc1);
    detector.detectAndCompute(image2, kpts2, desc2);
    matcher.match(kpts1, kpts2, desc1, desc2, matches);

    std::cout << "Initial matches: " << matches.size() << std::endl;

    applyLikelihoodBoundary(matches, params);

    std::cout << "Final correspondences: " << matches.size() << std::endl;

    return 0;
}