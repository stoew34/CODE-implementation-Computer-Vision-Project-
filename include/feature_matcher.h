#ifndef FEATURE_MATCHER_H
#define FEATURE_MATCHER_H

#include "types.h"
#include <opencv2/features2d.hpp>

namespace code {

class FeatureMatcher {
public:
    explicit FeatureMatcher(const CodeParameters& params);
    
    // Generate initial noisy matches using relaxed ratio test
    void match(const std::vector<cv::KeyPoint>& kpts1,
              const std::vector<cv::KeyPoint>& kpts2,
              const cv::Mat& desc1,
              const cv::Mat& desc2,
              std::vector<MatchHypothesis>& matches);
    
private:
    CodeParameters params_;
    cv::Ptr<cv::FlannBasedMatcher> matcher_;
    
    // Compute relative affine orientation between two keypoints
    cv::Vec4f computeRelativeOrientation(const cv::KeyPoint& kp1,
                                        const cv::KeyPoint& kp2);
    
    // Normalize coordinates using Hartley normalization
    void normalizeCoordinates(std::vector<MatchHypothesis>& matches,
                            cv::Vec2f& mean1, cv::Vec2f& mean2,
                            float& scale);
};

}

#endif // FEATURE_MATCHER_H
