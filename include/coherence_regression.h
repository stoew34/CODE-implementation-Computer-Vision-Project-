#ifndef COHERENCE_REGRESSION_H
#define COHERENCE_REGRESSION_H

#include "types.h"
#include <Eigen/Dense>
#include <vector>

namespace code {

// Base class for coherence-based regression
class CoherenceRegression {
public:
    explicit CoherenceRegression(const CodeParameters& params);
    virtual ~CoherenceRegression() = default;
    
protected:
    CodeParameters params_;
    
    // Cluster centroids for approximation
    std::vector<BilateralPoint8D> centroids_;
    
    // Gram matrix
    Eigen::MatrixXf G_;
    
    // Compute RBF kernel (Gaussian)
    float rbfKernel(const BilateralPoint8D& p1, const BilateralPoint8D& p2) const;
    
    // Huber loss function
    float huberLoss(float residual) const;
    
    // Gradient of Huber loss
    float huberGradient(float residual) const;
    
    // Perform K-means clustering to get representative points
    void clusterPoints(const std::vector<BilateralPoint8D>& points,
                      int num_clusters);
    
    // Build Gram matrix from centroids
    void buildGramMatrix();
    
    // Evaluate regression function at a point
    float evaluateFunction(const BilateralPoint8D& point,
                          const Eigen::VectorXf& weights,
                          float bias = 0.0f) const;
};

}

#endif // COHERENCE_REGRESSION_H
