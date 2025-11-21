#include "coherence_regression.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <random>

namespace code {

CoherenceRegression::CoherenceRegression(const CodeParameters& params)
    : params_(params) {
}

float CoherenceRegression::rbfKernel(const BilateralPoint8D& p1,
                                    const BilateralPoint8D& p2) const {
    float dist_sq = p1.distanceSquared(p2);
    float gamma_sq = params_.gamma * params_.gamma;
    return std::exp(-dist_sq / gamma_sq);
}

float CoherenceRegression::huberLoss(float residual) const {
    float abs_res = std::abs(residual);
    if (abs_res <= params_.epsilon) {
        return residual * residual;
    } else {
        return 2.0f * params_.epsilon * abs_res - params_.epsilon * params_.epsilon;
    }
}

float CoherenceRegression::huberGradient(float residual) const {
    if (std::abs(residual) <= params_.epsilon) {
        return 2.0f * residual;
    } else {
        return 2.0f * params_.epsilon * (residual > 0 ? 1.0f : -1.0f);
    }
}

void CoherenceRegression::clusterPoints(const std::vector<BilateralPoint8D>& points,
                                       int num_clusters) {
    if (points.empty()) return;
    
    // Use OpenCV K-means on 8D points
    int n = std::min(static_cast<int>(points.size()), params_.max_samples);
    cv::Mat data(n, 8, CV_32F);
    
    for (int i = 0; i < n; ++i) {
        data.at<float>(i, 0) = points[i].x;
        data.at<float>(i, 1) = points[i].y;
        data.at<float>(i, 2) = points[i].u;
        data.at<float>(i, 3) = points[i].v;
        data.at<float>(i, 4) = points[i].o1;
        data.at<float>(i, 5) = points[i].o2;
        data.at<float>(i, 6) = points[i].o3;
        data.at<float>(i, 7) = points[i].o4;
    }
    
    cv::Mat labels, centers;
    int K = std::min(num_clusters, n);
    
    cv::kmeans(data, K, labels,
              cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-4),
              3, cv::KMEANS_PP_CENTERS, centers);
    
    // Convert centers to centroids
    centroids_.clear();
    centroids_.reserve(K);
    for (int i = 0; i < K; ++i) {
        BilateralPoint8D c;
        c.x = centers.at<float>(i, 0);
        c.y = centers.at<float>(i, 1);
        c.u = centers.at<float>(i, 2);
        c.v = centers.at<float>(i, 3);
        c.o1 = centers.at<float>(i, 4);
        c.o2 = centers.at<float>(i, 5);
        c.o3 = centers.at<float>(i, 6);
        c.o4 = centers.at<float>(i, 7);
        centroids_.push_back(c);
    }
    
    std::cout << "Clustered " << n << " points into " << K << " centroids" << std::endl;
}

void CoherenceRegression::buildGramMatrix() {
    int M = centroids_.size();
    G_.resize(M, M);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            G_(i, j) = rbfKernel(centroids_[i], centroids_[j]);
        }
    }
}

float CoherenceRegression::evaluateFunction(const BilateralPoint8D& point,
                                           const Eigen::VectorXf& weights,
                                           float bias) const {
    float result = bias;
    int M = centroids_.size();
    
    for (int i = 0; i < M; ++i) {
        result += weights(i) * rbfKernel(point, centroids_[i]);
    }
    
    return result;
}

}
