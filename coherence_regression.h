#pragma once

#include "types.h"
#include "config.h"
#include "bilateral_clusterer.h"

namespace code {

    class CoherenceRegression {
    public:
        explicit CoherenceRegression(const Config& config);

        // Compute likelihood function f~(p) (Eqn 20-21)
        RegressionResult computeLikelihoodFunction(
            const std::vector<BilateralPoint8D>& points);

        // Evaluate likelihood at point p (Eqn 20)
        double evaluateLikelihood(
            const BilateralPoint8D& point,
            const RegressionResult& result,
            const std::vector<BilateralPoint8D>& centroids);

        // Apply likelihood filter (Eqn 22)
        std::vector<bool> filterByLikelihood(
            const std::vector<BilateralPoint8D>& points,
            const RegressionResult& result,
            const std::vector<BilateralPoint8D>& centroids);

    private:
        Config config_;
        BilateralClusterer clusterer_;

        // Huber loss function C(·) (Eqn 6)
        struct HuberLoss {
            double a;
            explicit HuberLoss(double a_) : a(a_) {}

            template<typename T>
            bool operator()(const T* residual, T* loss) const {
                T abs_r = ceres::abs(*residual);
                if (abs_r < T(a)) {
                    *loss = residual[0] * residual[0];
                }
                else {
                    *loss = T(2.0 * a) * abs_r - T(a * a);
                }
                return true;
            }
        };

        // Build Ceres problem for likelihood regression
        void buildLikelihoodProblem(
            ceres::Problem& problem,
            const Eigen::MatrixXd& basis_matrix,
            const Eigen::VectorXd& targets,
            Eigen::VectorXd& weights,
            const Eigen::MatrixXd& kernel_matrix);
    };

} // namespace code