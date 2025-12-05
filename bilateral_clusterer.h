#pragma once

#include "types.h"
#include "config.h"

namespace code {

    class BilateralClusterer {
    public:
        explicit BilateralClusterer(const Config& config);

        // Cluster points using k-means (Sec 2.2)
        std::vector<BilateralPoint8D> cluster(
            const std::vector<BilateralPoint8D>& points) const;

        // Compute Gaussian kernel matrix G~ (Eqn 11)
        Eigen::MatrixXd computeKernelMatrix(
            const std::vector<BilateralPoint8D>& centroids) const;

        // Compute basis functions g(p_i, p~_j) (Eqn 14)
        Eigen::MatrixXd computeBasisMatrix(
            const std::vector<BilateralPoint8D>& points,
            const std::vector<BilateralPoint8D>& centroids) const;

        // Compute basis for single point
        Eigen::VectorXd computeBasisForPoint(
            const BilateralPoint8D& point,
            const std::vector<BilateralPoint8D>& centroids) const;

    private:
        Config config_;

        // Gaussian kernel g(p_i, p_j) = exp(-||p_i - p_j||^2/γ^2) (Eqn 11)
        double gaussianKernel(const BilateralPoint8D& p1,
            const BilateralPoint8D& p2) const;

        // Initialize centroids with k-means++
        void initializeCentroids(const std::vector<BilateralPoint8D>& points,
            std::vector<BilateralPoint8D>& centroids) const;

        // Assign points to nearest centroid
        void assignToCentroids(const std::vector<BilateralPoint8D>& points,
            const std::vector<BilateralPoint8D>& centroids,
            std::vector<int>& assignments) const;
    };

} // namespace code