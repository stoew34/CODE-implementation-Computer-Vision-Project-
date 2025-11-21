#ifndef TYPES_H
#define TYPES_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace code {

// 8D bilateral point structure: spatial (x,y) + motion (u,v) + orientation (o1-o4)
struct BilateralPoint8D {
    float x, y;           // Spatial coordinates
    float u, v;           // Motion vectors
    float o1, o2, o3, o4; // Relative affine orientation
    
    BilateralPoint8D() : x(0), y(0), u(0), v(0), o1(0), o2(0), o3(0), o4(0) {}
    
    BilateralPoint8D(float x_, float y_, float u_, float v_,
                     float o1_, float o2_, float o3_, float o4_)
        : x(x_), y(y_), u(u_), v(v_), o1(o1_), o2(o2_), o3(o3_), o4(o4_) {}
    
    // Compute squared distance in bilateral space
    float distanceSquared(const BilateralPoint8D& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float du = u - other.u;
        float dv = v - other.v;
        float do1 = o1 - other.o1;
        float do2 = o2 - other.o2;
        float do3 = o3 - other.o3;
        float do4 = o4 - other.o4;
        return dx*dx + dy*dy + du*du + dv*dv + do1*do1 + do2*do2 + do3*do3 + do4*do4;
    }
};

// Match hypothesis structure
struct MatchHypothesis {
    cv::KeyPoint kp1;           // Keypoint in image 1
    cv::KeyPoint kp2;           // Keypoint in image 2
    cv::Vec2f motion;           // Motion vector (u, v)
    cv::Vec4f orientation;      // Relative orientation parameters
    
    BilateralPoint8D point_likelihood;  // Point for likelihood boundary
    BilateralPoint8D point_affine_x;    // Point for X affine boundary
    BilateralPoint8D point_affine_y;    // Point for Y affine boundary
    
    float likelihood_score;     // Score from likelihood boundary
    float affine_error;         // Error from affine boundary
    bool is_inlier;            // Final inlier status
    
    MatchHypothesis() : likelihood_score(0), affine_error(0), is_inlier(false) {}
};

// Parameters structure
struct CodeParameters {
    // Feature detection
    int num_affine_samples = 7;    // Number of affine warps for A-SIFT simulation
    
    // Matching
    float match_ratio = 0.9f;      // Relaxed ratio test threshold
    
    // Regression
    float lambda = 1.0f;           // Smoothness weight
    float gamma = 1.0f;            // RBF kernel width
    float epsilon = 0.1f;          // Huber loss threshold
    int num_clusters = 100;        // Number of K-means clusters (M)
    int max_samples = 30000;       // Max samples for likelihood (N)
    int max_affine_samples = 1000; // Max samples for affine regression
    
    // Filtering
    float likelihood_threshold = 0.6f;  // Likelihood acceptance threshold
    float spatial_threshold = 0.01f;    // Affine spatial error threshold (normalized)
    
    // Optimization
    int max_iterations = 100;
    float convergence_tolerance = 1e-6f;
    
    CodeParameters() {}
};

}

#endif // TYPES_H
