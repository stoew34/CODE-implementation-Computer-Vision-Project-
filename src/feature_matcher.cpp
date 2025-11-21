#include "feature_matcher.h"
#include <iostream>

namespace code {

FeatureMatcher::FeatureMatcher(const CodeParameters& params)
    : params_(params) {
    matcher_ = cv::FlannBasedMatcher::create();
}

cv::Vec4f FeatureMatcher::computeRelativeOrientation(const cv::KeyPoint& kp1,
                                                     const cv::KeyPoint& kp2) {
    // Compute relative affine parameters from keypoint properties
    float angle_diff = kp2.angle - kp1.angle;
    float scale_ratio = kp2.size / std::max(kp1.size, 1e-6f);
    
    // Simple 4D orientation encoding
    cv::Vec4f orientation;
    orientation[0] = std::cos(angle_diff * CV_PI / 180.0f);
    orientation[1] = std::sin(angle_diff * CV_PI / 180.0f);
    orientation[2] = scale_ratio;
    orientation[3] = 1.0f / std::max(scale_ratio, 1e-6f);
    
    return orientation;
}

void FeatureMatcher::normalizeCoordinates(std::vector<MatchHypothesis>& matches,
                                         cv::Vec2f& mean1, cv::Vec2f& mean2,
                                         float& scale) {
    if (matches.empty()) return;
    
    // Compute mean
    mean1 = cv::Vec2f(0, 0);
    mean2 = cv::Vec2f(0, 0);
    for (const auto& m : matches) {
        mean1[0] += m.kp1.pt.x;
        mean1[1] += m.kp1.pt.y;
        mean2[0] += m.kp2.pt.x;
        mean2[1] += m.kp2.pt.y;
    }
    mean1 /= static_cast<float>(matches.size());
    mean2 /= static_cast<float>(matches.size());
    
    // Compute average distance from center
    float avg_dist = 0;
    for (const auto& m : matches) {
        float dx1 = m.kp1.pt.x - mean1[0];
        float dy1 = m.kp1.pt.y - mean1[1];
        avg_dist += std::sqrt(dx1*dx1 + dy1*dy1);
    }
    avg_dist /= matches.size();
    
    // Scale to make average distance = sqrt(2)
    scale = std::sqrt(2.0f) / std::max(avg_dist, 1e-6f);
    
    // Apply normalization
    for (auto& m : matches) {
        m.kp1.pt.x = (m.kp1.pt.x - mean1[0]) * scale;
        m.kp1.pt.y = (m.kp1.pt.y - mean1[1]) * scale;
        m.kp2.pt.x = (m.kp2.pt.x - mean2[0]) * scale;
        m.kp2.pt.y = (m.kp2.pt.y - mean2[1]) * scale;
        
        // Update motion
        m.motion[0] = m.kp2.pt.x - m.kp1.pt.x;
        m.motion[1] = m.kp2.pt.y - m.kp1.pt.y;
    }
}

void FeatureMatcher::match(const std::vector<cv::KeyPoint>& kpts1,
                          const std::vector<cv::KeyPoint>& kpts2,
                          const cv::Mat& desc1,
                          const cv::Mat& desc2,
                          std::vector<MatchHypothesis>& matches) {
    matches.clear();
    
    // K-nearest neighbor matching with k=2
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(desc1, desc2, knn_matches, 2);
    
    // Apply relaxed ratio test
    for (const auto& knn : knn_matches) {
        if (knn.size() < 2) continue;
        
        float ratio = knn[0].distance / knn[1].distance;
        if (ratio < params_.match_ratio) {
            MatchHypothesis m;
            m.kp1 = kpts1[knn[0].queryIdx];
            m.kp2 = kpts2[knn[0].trainIdx];
            m.motion[0] = m.kp2.pt.x - m.kp1.pt.x;
            m.motion[1] = m.kp2.pt.y - m.kp1.pt.y;
            m.orientation = computeRelativeOrientation(m.kp1, m.kp2);
            
            matches.push_back(m);
        }
    }
    
    std::cout << "Generated " << matches.size() << " initial matches with ratio < " 
              << params_.match_ratio << std::endl;
    
    // Apply Hartley normalization
    cv::Vec2f mean1, mean2;
    float scale;
    normalizeCoordinates(matches, mean1, mean2, scale);
    
    std::cout << "Applied Hartley normalization (scale: " << scale << ")" << std::endl;
}

}
