#pragma once

#include "types.h"
#include "config.h"
#include "bilateral_clusterer.h"

namespace code {

    class BilateralAffine {
    public:
        explicit BilateralAffine(const Config& config);

        // Compute bilaterally varying affine models (Eqns 24-27)
        std::vector<RegressionResult> computeAffineModels(
            const std::vector<BilateralPoint8D>& points,
            const std::vector<BilateralPoint8D>& centroids);

        // Evaluate affine prediction for X direction (Eqn 24)
        double evaluateAffineX(
            const BilateralPoint8D& point,
            const std::vector<RegressionResult>& models,
            const std::vector<BilateralPoint8D>& centroids);

        // Evaluate affine prediction for Y direction (Eqn 26)
        double evaluateAffineY(
            const BilateralPoint8D& point,
            const std::vector<RegressionResult>& models,
            const std::vector<BilateralPoint8D>& centroids);

        // Compute spatial residual (for Eqn 28)
        double computeSpatialResidual(
            const BilateralPoint8D& point,
            const std::vector<RegressionResult>& models,
            const std::vector<BilateralPoint8D>& centroids);

        // Apply spatial filter (Eqn 28)
        std::vector<bool> filterBySpatialConsistency(
            const std::vector<BilateralPoint8D>& points,
            const std::vector<RegressionResult>& models,
            const std::vector<BilateralPoint8D>& centroids);

    private:
        Config config_;
        BilateralClusterer clusterer_;

        // Compute affine basis: a1(p)=x, a2(p)=y, a3(p)=1
        Eigen::Vector3d computeAffineBasis(const BilateralPoint8D& point) const;

        // Build Ceres problem for affine regression
        void buildAffineProblem(
            ceres::Problem& problem,
            const Eigen::MatrixXd& basis_matrix,
            const std::vector<BilateralPoint8D>& points,
            Eigen::MatrixXd& affine_weights,  // 6 x M matrix
            const Eigen::MatrixXd& kernel_matrix,
            int direction);  // 0 for X, 1 for Y
    };

} // namespace code