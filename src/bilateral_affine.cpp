#include "bilateral_affine.h"
#include <algorithm>
#include <numeric>

namespace code {

    // Helper cost function classes
    // Cost for one point fitting all 3 affine parameters
    struct AffinePointCost {
        std::vector<double> basis_vals;  // RBF basis for this point
        Eigen::Vector3d affine_basis;    // (x, y, 1)
        double target_val;
        int num_centroids;

        AffinePointCost(const std::vector<double>& basis, const Eigen::Vector3d& affine,
                        double target, int m)
            : basis_vals(basis), affine_basis(affine), target_val(target), num_centroids(m) {
        }

        // Cost function: target - sum_i(w[param][i] * basis[i] * affine_basis[param])
        // This takes 3*m parameters: w[0][0..m-1], w[1][0..m-1], w[2][0..m-1]
        bool operator()(double const* const* weights, double* residual) const {
            double prediction = 0.0;
            for (int param = 0; param < 3; ++param) {
                double f_param = 0.0;
                for (int i = 0; i < num_centroids; ++i) {
                    f_param += weights[param][i] * basis_vals[i];
                }
                prediction += f_param * affine_basis(param);
            }
            residual[0] = target_val - prediction;
            return true;
        }
    };

    struct RegularizationCost {
        double weight;
        int idx1, idx2;  // Indices within the parameter block

        RegularizationCost(double w, int i, int j) : weight(w), idx1(i), idx2(j) {}

        // This operates on full m-sized parameter blocks
        // For DynamicNumericDiffCostFunction, signature must be (double const* const*, double*)
        bool operator()(double const* const* weights, double* residual) const {
            residual[0] = weight * (weights[0][idx1] - weights[0][idx2]);
            return true;
        }
    };

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
            int m = centroids.size();

            // Use separate contiguous arrays for each parameter
            std::vector<double> weights_param0(m, 0.0);
            std::vector<double> weights_param1(m, 0.0);
            std::vector<double> weights_param2(m, 0.0);

            ceres::Problem problem;
            buildAffineProblem(problem, basis_matrix, points,
                weights_param0, weights_param1, weights_param2,
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

                // Copy weights to Eigen vector
                Eigen::VectorXd weight_vec(m);
                const std::vector<double>& src = (param == 0) ? weights_param0 :
                                                  (param == 1) ? weights_param1 : weights_param2;
                for (int i = 0; i < m; ++i) {
                    weight_vec(i) = src[i];
                }

                results[idx].weights = weight_vec;
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
        std::vector<double> all_residuals;

        // Debug first few points
        int debug_count = std::min(5, (int)points.size());
        for (int i = 0; i < debug_count; i++) {
            logInfo("DEBUG Point " + std::to_string(i) + ": x=" + std::to_string(points[i].x) +
                    ", y=" + std::to_string(points[i].y) + ", u=" + std::to_string(points[i].u) +
                    ", v=" + std::to_string(points[i].v));
        }

        for (size_t i = 0; i < points.size(); ++i) {
            double residual = computeSpatialResidual(points[i], models, centroids);
            all_residuals.push_back(residual);
            inliers[i] = (residual < config_.spatial_threshold);

            // Debug first few residuals
            if (i < 5) {
                logInfo("DEBUG Residual " + std::to_string(i) + ": " + std::to_string(residual) +
                        " (threshold=" + std::to_string(config_.spatial_threshold) + ")");
            }
        }

        // Debug: print residual statistics
        if (!all_residuals.empty()) {
            double min_res = *std::min_element(all_residuals.begin(), all_residuals.end());
            double max_res = *std::max_element(all_residuals.begin(), all_residuals.end());
            double avg_res = std::accumulate(all_residuals.begin(), all_residuals.end(), 0.0) / all_residuals.size();
            int num_inliers = std::count(inliers.begin(), inliers.end(), true);

            // Calculate stats for inliers only
            std::vector<double> inlier_residuals;
            for (size_t i = 0; i < all_residuals.size(); ++i) {
                if (inliers[i]) {
                    inlier_residuals.push_back(all_residuals[i]);
                }
            }

            if (!inlier_residuals.empty()) {
                double inlier_min = *std::min_element(inlier_residuals.begin(), inlier_residuals.end());
                double inlier_max = *std::max_element(inlier_residuals.begin(), inlier_residuals.end());
                double inlier_avg = std::accumulate(inlier_residuals.begin(), inlier_residuals.end(), 0.0) / inlier_residuals.size();

                logInfo("Spatial residuals (all) - min: " + std::to_string(min_res) +
                        ", max: " + std::to_string(max_res) +
                        ", avg: " + std::to_string(avg_res));
                logInfo("Spatial residuals (inliers only) - min: " + std::to_string(inlier_min) +
                        ", max: " + std::to_string(inlier_max) +
                        ", avg: " + std::to_string(inlier_avg) +
                        ", count: " + std::to_string(num_inliers) + "/" + std::to_string(points.size()));
            }
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
        std::vector<double>& weights_param0,
        std::vector<double>& weights_param1,
        std::vector<double>& weights_param2,
        const Eigen::MatrixXd& kernel_matrix,
        int direction) {

        int n = points.size();
        int m = basis_matrix.cols();

        // For each point, add one residual block that uses all parameters
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d affine_basis = computeAffineBasis(points[i]);
            double target = (direction == 0) ?
                (points[i].x + points[i].u) :
                (points[i].y + points[i].v);

            // Extract basis values for this point
            std::vector<double> basis_vals(m);
            for (int j = 0; j < m; ++j) {
                basis_vals[j] = basis_matrix(i, j);
            }

            // Create parameter block pointers for all 3 affine parameters
            std::vector<double*> parameter_blocks = {
                weights_param0.data(),
                weights_param1.data(),
                weights_param2.data()
            };

            // Create cost function that takes 3 parameter blocks, each of size m
            ceres::DynamicNumericDiffCostFunction<AffinePointCost>* cost_function =
                new ceres::DynamicNumericDiffCostFunction<AffinePointCost>(
                    new AffinePointCost(basis_vals, affine_basis, target, m));

            cost_function->SetNumResiduals(1);
            for (int param = 0; param < 3; ++param) {
                cost_function->AddParameterBlock(m);
            }

            problem.AddResidualBlock(cost_function,
                new ceres::HuberLoss(config_.huber_epsilon),
                parameter_blocks);
        }

        // Add regularization for each parameter
        double* param_arrays[3] = {
            weights_param0.data(),
            weights_param1.data(),
            weights_param2.data()
        };

        for (int param = 0; param < 3; ++param) {
            for (int i = 0; i < m; ++i) {
                for (int j = i + 1; j < m; ++j) {
                    if (kernel_matrix(i, j) != 0) {
                        double reg_weight = config_.lambda * kernel_matrix(i, j);

                        // Create cost that operates on one m-sized parameter block
                        ceres::DynamicNumericDiffCostFunction<RegularizationCost>* reg_cost =
                            new ceres::DynamicNumericDiffCostFunction<RegularizationCost>(
                                new RegularizationCost(reg_weight, i, j));

                        reg_cost->SetNumResiduals(1);
                        reg_cost->AddParameterBlock(m);

                        problem.AddResidualBlock(reg_cost, nullptr, param_arrays[param]);
                    }
                }
            }
        }
    }

} // namespace code