#include "bilateral_affine.h"

namespace code {

    BilateralAffine::BilateralAffine(const Config& config)
        : config_(config), clusterer_(config) {
    }

    std::vector<RegressionResult> BilateralAffine::computeAffineModels(
        const std::vector<BilateralPoint8D>& points,
        const std::vector<BilateralPoint8D>& centroids) {

        std::vector<RegressionResult> results(6);  // 6 affine parameters

        if (points.empty() || centroids.empty()) {
            logWarning("No points or centroids for affine regression");
            return results;
        }

        // Compute basis matrix
        auto basis_matrix = clusterer_.computeBasisMatrix(points, centroids);
        auto kernel_matrix = clusterer_.computeKernelMatrix(centroids);

        // Build and solve for X and Y directions
        for (int dir = 0; dir < 2; ++dir) {  // 0: X, 1: Y
            Eigen::MatrixXd affine_weights = Eigen::MatrixXd::Zero(3, centroids.size());

            ceres::Problem problem;
            buildAffineProblem(problem, basis_matrix, points, affine_weights,
                kernel_matrix, dir);

            // Solve with Ceres
            ceres::Solver::Options options;
            options.max_num_iterations = config_.max_iterations;
            options.function_tolerance = config_.function_tolerance;
            options.minimizer_progress_to_stdout = false;
            options.num_threads = config_.num_threads;
            options.use_nonmonotonic_steps = config_.use_nonmonotonic_steps;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // Store results for each parameter
            for (int param = 0; param < 3; ++param) {
                int idx = dir * 3 + param;
                results[idx].weights = affine_weights.row(param);
                results[idx].basis_matrix = basis_matrix;
                results[idx].final_cost = summary.final_cost / 6.0;
                results[idx].converged = (summary.termination_type == ceres::CONVERGENCE);
            }
        }

        return results;
    }

    double BilateralAffine::evaluateAffineX(
        const BilateralPoint8D& point,
        const std::vector<RegressionResult>& models,
        const std::vector<BilateralPoint8D>& centroids) {

        if (models.size() < 6 || centroids.empty()) {
            return 0.0;
        }

        Eigen::VectorXd basis = clusterer_.computeBasisForPoint(point, centroids);
        Eigen::Vector3d affine_basis = computeAffineBasis(point);

        // f_x(p) = f1(p)*x + f2(p)*y + f3(p) (Eqn 24)
        double result = 0.0;
        for (int i = 0; i < 3; ++i) {
            double f_i = basis.dot(models[i].weights);
            result += f_i * affine_basis(i);
        }

        return result;
    }

    double BilateralAffine::evaluateAffineY(
        const BilateralPoint8D& point,
        const std::vector<RegressionResult>& models,
        const std::vector<BilateralPoint8D>& centroids) {

        if (models.size() < 6 || centroids.empty()) {
            return 0.0;
        }

        Eigen::VectorXd basis = clusterer_.computeBasisForPoint(point, centroids);
        Eigen::Vector3d affine_basis = computeAffineBasis(point);

        // f_y(p) = f4(p)*x + f5(p)*y + f6(p) (Eqn 26)
        double result = 0.0;
        for (int i = 3; i < 6; ++i) {
            double f_i = basis.dot(models[i].weights);
            result += f_i * affine_basis(i - 3);
        }

        return result;
    }

    double BilateralAffine::computeSpatialResidual(
        const BilateralPoint8D& point,
        const std::vector<RegressionResult>& models,
        const std::vector<BilateralPoint8D>& centroids) {

        double pred_x = evaluateAffineX(point, models, centroids);
        double pred_y = evaluateAffineY(point, models, centroids);

        double residual_x = pred_x - (point.x + point.u);
        double residual_y = pred_y - (point.y + point.v);

        return std::sqrt(residual_x * residual_x + residual_y * residual_y);
    }

    std::vector<bool> BilateralAffine::filterBySpatialConsistency(
        const std::vector<BilateralPoint8D>& points,
        const std::vector<RegressionResult>& models,
        const std::vector<BilateralPoint8D>& centroids) {

        std::vector<bool> inliers(points.size(), false);

        for (size_t i = 0; i < points.size(); ++i) {
            double residual = computeSpatialResidual(points[i], models, centroids);
            inliers[i] = (residual < config_.spatial_threshold);
        }

        return inliers;
    }

    Eigen::Vector3d BilateralAffine::computeAffineBasis(const BilateralPoint8D& point) const {
        Eigen::Vector3d basis;
        basis << point.x, point.y, 1.0;
        return basis;
    }

    void BilateralAffine::buildAffineProblem(
        ceres::Problem& problem,
        const Eigen::MatrixXd& basis_matrix,
        const std::vector<BilateralPoint8D>& points,
        Eigen::MatrixXd& affine_weights,
        const Eigen::MatrixXd& kernel_matrix,
        int direction) {

        int n = points.size();
        int m = basis_matrix.cols();

        // For each point, add residual terms
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d affine_basis = computeAffineBasis(points[i]);
            double target = (direction == 0) ?
                (points[i].x + points[i].u) :
                (points[i].y + points[i].v);

            for (int param = 0; param < 3; ++param) {
                for (int j = 0; j < m; ++j) {
                    double basis_val = basis_matrix(i, j);
                    if (basis_val > 1e-6) {
                        // Create cost function for this term
                        ceres::CostFunction* cost_function =
                            new ceres::NumericDiffCostFunction<AffineTermCost,
                            ceres::CENTRAL, 1, 1>(
                                new AffineTermCost(basis_val, affine_basis(param), target));

                        problem.AddResidualBlock(cost_function,
                            new ceres::HuberLoss(config_.huber_epsilon),
                            &affine_weights(param, j));
                    }
                }
            }
        }

        // Add regularization for each parameter
        for (int param = 0; param < 3; ++param) {
            for (int i = 0; i < m; ++i) {
                for (int j = i; j < m; ++j) {
                    if (kernel_matrix(i, j) != 0) {
                        double reg_weight = config_.lambda * kernel_matrix(i, j);

                        ceres::CostFunction* reg_cost =
                            new ceres::NumericDiffCostFunction<RegularizationCost,
                            ceres::CENTRAL, 1, 1, 1>(
                                new RegularizationCost(reg_weight));

                        problem.AddResidualBlock(reg_cost, nullptr,
                            &affine_weights(param, i),
                            &affine_weights(param, j));
                    }
                }
            }
        }
    }

    // Helper cost function for affine terms
    struct AffineTermCost {
        double basis_val, affine_basis_val, target_val;

        AffineTermCost(double basis, double affine_basis, double target)
            : basis_val(basis), affine_basis_val(affine_basis), target_val(target) {
        }

        bool operator()(const double* const w, double* residual) const {
            residual[0] = target_val - basis_val * affine_basis_val * w[0];
            return true;
        }
    };

} // namespace code