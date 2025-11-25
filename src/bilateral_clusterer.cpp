#include "bilateral_clusterer.h"
#include <iostream>
#include <algorithm>
#include <random>

namespace code {

BilateralClusterer::BilateralClusterer(const CodeParameters& params)
    : params_(params) {
}

float BilateralClusterer::rbfKernel(const BilateralPoint8D& p1,
                                   const BilateralPoint8D& p2) const {
    float dist_sq = p1.distanceSquared(p2);
    float gamma_sq = params_.gamma * params_.gamma;
    return std::exp(-dist_sq / gamma_sq);
}

void BilateralClusterer::buildGramMatrix() {
    int M = centroids_.size();
    gram_matrix_.resize(M, M);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            gram_matrix_(i, j) = rbfKernel(centroids_[i], centroids_[j]);
        }
    }
    
    std::cout << "Built Gram matrix: " << M << "×" << M << std::endl;
}

void BilateralClusterer::cluster(const std::vector<MatchHypothesis>& matches,
                                int num_clusters,
                                int max_samples) {
    if (matches.empty()) {
        std::cout << "Warning: No matches to cluster!" << std::endl;
        return;
    }
    
    // Subsample if too many matches
    int n = std::min(static_cast<int>(matches.size()), max_samples);
    
    // Prepare data for K-means (8D points)
    cv::Mat data(n, 8, CV_32F);
    for (int i = 0; i < n; ++i) {
        const auto& m = matches[i];
        data.at<float>(i, 0) = m.kp1.pt.x;
        data.at<float>(i, 1) = m.kp1.pt.y;
        data.at<float>(i, 2) = m.motion[0];
        data.at<float>(i, 3) = m.motion[1];
        data.at<float>(i, 4) = m.orientation[0];
        data.at<float>(i, 5) = m.orientation[1];
        data.at<float>(i, 6) = m.orientation[2];
        data.at<float>(i, 7) = m.orientation[3];
    }
    
    // Run K-means
    cv::Mat labels, centers;
    int K = std::min(num_clusters, n);
    
    std::cout << "Clustering " << n << " points into " << K << " centroids..." << std::endl;
    
    cv::kmeans(data, K, labels,
              cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-4),
              3, cv::KMEANS_PP_CENTERS, centers);
    
    // Convert centers to BilateralPoint8D
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
    
    // Build Gram matrix
    buildGramMatrix();
    
    std::cout << "Clustering complete: " << K << " centroids" << std::endl;
}

} // namespace code
