#include "coherence_regression.h"
#include <iostream>

namespace code {

    void LikelihoodSolver::buildGramMatrix() {
        int M = centroids_.size();
        G_.resize(M, M);

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                G_(i, j) = rbfKernel(centroids_[i], centroids_[j]);
            }
        }
    }

    float LikelihoodSolver::rbfKernel(const BilateralPoint8D& p1, const BilateralPoint8D& p2) const {
        float dist_sq = p1.distanceSquared(p2);
        float gamma_sq = params_.gamma * params_.gamma;
        return std::exp(-dist_sq / gamma_sq);
    }

    void LikelihoodSolver::solve(const std::vector<MatchHypothesis>& matches) {
        int M = centroids_.size();
        int N = matches.size();

        if (M == 0 || N == 0) {
            std::cout << "Warning: No centroids or matches for likelihood solver" << std::endl;
            return;
        }

        // Initialize weights to zero
        std::vector<double> weights(M, 0.0);

        // Convert matches to bilateral points
        std::vector<BilateralPoint8D> points;
        points.reserve(N);
        for (const auto& match : matches) {
            BilateralPoint8D p;
            p.x = match.kp1.pt.x;
            p.y = match.kp1.pt.y;
            p.u = match.motion[0];
            p.v = match.motion[1];
            p.o1 = match.orientation[0];
            p.o2 = match.orientation[1];
            p.o3 = match.orientation[2];
            p.o4 = match.orientation[3];
            points.push_back(p);
        }

        std::cout << "Starting Ceres optimization: " << N << " points, "
            << M << " centroids" << std::endl;

        // Build Ceres problem
        ceres::Problem problem;

        // Add data terms with Huber loss
        for (const auto& point : points) {
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<LikelihoodDataCost, 1, Eigen::Dynamic>(
                    new LikelihoodDataCost(point, centroids_, params_.gamma), M);

            problem.AddResidualBlock(cost_function,
                new ceres::HuberLoss(params_.epsilon),
                weights.data());
        }

        // Add smoothness term
        ceres::CostFunction* smoothness_cost =
            new ceres::AutoDiffCostFunction<SmoothnessCost, 1, Eigen::Dynamic>(
                new SmoothnessCost(G_, params_.lambda, M), M);
        problem.AddResidualBlock(smoothness_cost, nullptr, weights.data());

        // Configure solver
        ceres::Solver::Options options;
        options.max_num_iterations = 100;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.num_threads = 8;

        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.BriefReport() << std::endl;

        // Convert weights back to Eigen
        weights_ = Eigen::VectorXf(M);
        for (int i = 0; i < M; ++i) {
            weights_(i) = static_cast<float>(weights[i]);
        }

        std::cout << "Likelihood optimization complete. Final cost: "
            << summary.final_cost << std::endl;
    }

    float LikelihoodSolver::evaluate(const BilateralPoint8D& point) const {
        float result = 0.0f;
        int M = centroids_.size();

        for (int i = 0; i < M; ++i) {
            result += weights_(i) * rbfKernel(point, centroids_[i]);
        }

        return result;
    }

} // namespace code