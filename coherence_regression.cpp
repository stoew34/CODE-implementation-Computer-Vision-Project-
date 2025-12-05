#include "coherence_regression.h"

namespace code {

    CoherenceRegression::CoherenceRegression(const Config& config)
        : config_(config), clusterer_(config) {
    }

    RegressionResult CoherenceRegression::computeLikelihoodFunction(
        const std::vector<BilateralPoint8D>& points) {

        RegressionResult result;

        if (points.empty()) {
            logWarning("No points for likelihood regression");
            return result;
        }

        // 1. Cluster points (Sec 2.2)
        auto centroids = clusterer_.cluster(points);

        // 2. Compute basis matrix (Eqn 14)
        auto basis_matrix = clusterer_.computeBasisMatrix(points, centroids);

        // 3. Compute kernel matrix G~ (Eqn 11)
        auto kernel_matrix = clusterer_.computeKernelMatrix(centroids);

        // 4. Prepare targets: q_hat_j = 1 (Eqn 19)
        Eigen::VectorXd targets = Eigen::VectorXd::Ones(points.size());

        // 5. Initialize weights
        Eigen::VectorXd weights = Eigen::VectorXd::Zero(centroids.size());

        // 6. Build Ceres problem (Eqn 21)
        ceres::Problem problem;
        buildLikelihoodProblem(problem, basis_matrix, targets, weights, kernel_matrix);

        // 7. Solve with Ceres
        ceres::Solver::Options options;
        options.max_num_iterations = config_.max_iterations;
        options.function_tolerance = config_.function_tolerance;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = config_.num_threads;
        options.use_nonmonotonic_steps = config_.use_nonmonotonic_steps;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 8. Store results
        result.weights = weights;
        result.basis_matrix = basis_matrix;
        result.final_cost = summary.final_cost;
        result.converged = (summary.termination_type == ceres::CONVERGENCE);

        if (!result.converged) {
            logWarning("Likelihood regression did not converge fully");
        }

        return result;
    }

    double CoherenceRegression::evaluateLikelihood(
        const BilateralPoint8D& point,
        const RegressionResult& result,
        const std::vector<BilateralPoint8D>& centroids) {

        if (centroids.empty() || result.weights.size() != centroids.size()) {
            return 0.0;
        }

        // Compute basis functions for this point
        Eigen::VectorXd basis = clusterer_.computeBasisForPoint(point, centroids);

        double likelihood = basis.dot(result.weights);

        return likelihood;
    }

    std::vector<bool> CoherenceRegression::filterByLikelihood(
        const std::vector<BilateralPoint8D>& points,
        const RegressionResult& result,
        const std::vector<BilateralPoint8D>& centroids) {

        std::vector<bool> inliers(points.size(), false);

        for (size_t i = 0; i < points.size(); ++i) {
            double likelihood = evaluateLikelihood(points[i], result, centroids);
            inliers[i] = (likelihood > config_.likelihood_threshold);
        }

        return inliers;
    }

    void CoherenceRegression::buildLikelihoodProblem(
        ceres::Problem& problem,
        const Eigen::MatrixXd& basis_matrix,
        const Eigen::VectorXd& targets,
        Eigen::VectorXd& weights,
        const Eigen::MatrixXd& kernel_matrix) {

        int n = basis_matrix.rows();
        int m = basis_matrix.cols();

        // Add data fitting terms with Huber loss
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                double basis_val = basis_matrix(i, j);
                if (basis_val > 1e-6) {
                    // Create cost function for this term
                    ceres::CostFunction* cost_function =
                        new ceres::NumericDiffCostFunction<BasisTermCost,
                        ceres::CENTRAL, 1, 1>(
                            new BasisTermCost(basis_val, targets(i)));

                    problem.AddResidualBlock(cost_function,
                        new ceres::HuberLoss(config_.huber_epsilon),
                        &weights(j));
                }
            }
        }

        for (int i = 0; i < m; ++i) {
            for (int j = i; j < m; ++j) {
                if (kernel_matrix(i, j) != 0) {
                    double reg_weight = config_.lambda * kernel_matrix(i, j);

                    // Add regularization term
                    ceres::CostFunction* reg_cost =
                        new ceres::NumericDiffCostFunction<RegularizationCost,
                        ceres::CENTRAL, 1, 1, 1>(
                            new RegularizationCost(reg_weight));

                    problem.AddResidualBlock(reg_cost, nullptr,
                        &weights(i), &weights(j));
                }
            }
        }
    }

    // Helper cost function classes
    struct BasisTermCost {
        double basis_val, target_val;

        BasisTermCost(double basis, double target)
            : basis_val(basis), target_val(target) {
        }

        bool operator()(const double* const w, double* residual) const {
            residual[0] = target_val - basis_val * w[0];
            return true;
        }
    };

    struct RegularizationCost {
        double weight;

        explicit RegularizationCost(double w) : weight(w) {}

        bool operator()(const double* const w1, const double* const w2,
            double* residual) const {
            residual[0] = weight * (w1[0] - w2[0]);
            return true;
        }
    };

} // namespace code