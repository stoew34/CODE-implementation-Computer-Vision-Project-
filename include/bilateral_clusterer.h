#ifndef BILATERAL_CLUSTERER_H
#define BILATERAL_CLUSTERER_H

#include "types.h"
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>

namespace code {

/**
 * @brief Clusters match hypotheses in 8D bilateral space
 * 
 * Used by both LikelihoodBoundary and AffineBoundary to get
 * representative centroids for RBF regression.
 */
class BilateralClusterer {
public:
    explicit BilateralClusterer(const CodeParameters& params);
    
    /**
     * @brief Cluster matches in 8D bilateral space using K-means
     * 
     * @param matches Input match hypotheses
     * @param num_clusters Number of clusters (M, typically 100)
     * @param max_samples Maximum samples to use (for speed)
     */
    void cluster(const std::vector<MatchHypothesis>& matches,
                int num_clusters = 100,
                int max_samples = 30000);
    
    /**
     * @brief Get cluster centroids in 8D space
     * @return Vector of M centroids [x, y, u, v, o1-o4]
     */
    const std::vector<BilateralPoint8D>& getCentroids() const {
        return centroids_;
    }
    
    /**
     * @brief Get Gram matrix for regression
     * @return M×M Gram matrix where G(i,j) = exp(-dist²(ci,cj)/γ²)
     */
    const Eigen::MatrixXf& getGramMatrix() const {
        return gram_matrix_;
    }
    
    /**
     * @brief Get number of centroids
     */
    int getNumCentroids() const {
        return centroids_.size();
    }
    
private:
    CodeParameters params_;
    std::vector<BilateralPoint8D> centroids_;
    Eigen::MatrixXf gram_matrix_;
    
    /**
     * @brief Build Gram matrix from centroids
     */
    void buildGramMatrix();
    
    /**
     * @brief Compute RBF kernel between two points
     */
    float rbfKernel(const BilateralPoint8D& p1, 
                   const BilateralPoint8D& p2) const;
};

} // namespace code

#endif // BILATERAL_CLUSTERER_H
