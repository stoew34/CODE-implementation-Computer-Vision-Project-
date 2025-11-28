#ifndef COHERENCE_REGRESSION_H
#define COHERENCE_REGRESSION_H

#include "bilateral_clusterer.h"
#include <ceres/ceres.h>
#include <vector>
#include <Eigen/Dense>

namespace code {

    struct LikelihoodDataCost {
        BilateralPoint8D point;
        std::vector<BilateralPoint8D> centroids;
        float gamma;

        LikelihoodDataCost(BilateralPoint8D p, std::vector<BilateralPoint8D> c, float g)
            : point(p), centroids(std::move(c)), gamma(g) {
        }

        template <typename T>
        bool operator()(const T* const weights, T* residual) const {
            T prediction = T(0);
            int M = centroids.size();

            for (int i = 0; i < M; ++i) {
                // Compute Gaussian kernel
                T dx = T(point.x) - T(centroids[i].x);
                T dy = T(point.y) - T(centroids[i].y);
                T du = T(point.u) - T(centroids[i].u);
                T dv = T(point.v) - T(centroids[i].v);
                T do1 = T(point.o1) - T(centroids[i].o1);
                T do2 = T(point.o2) - T(centroids[i].o2);
                T do3 = T(point.o3) - T(centroids[i].o3);
                T do4 = T(point.o4) - T(centroids[i].o4);

                T dist_sq = dx * dx + dy * dy + du * du + dv * dv + do1 * do1 + do2 * do2 + do3 * do3 + do4 * do4;
                T gamma_sq = T(gamma * gamma);
                T kernel_val = ceres::exp(-dist_sq / gamma_sq);

                prediction += T(weights[i]) * kernel_val;
            }

            residual[0] = T(1.0) - prediction;
            return true;
        }
    };

    // Cost function for smoothness term
    struct SmoothnessCost {
        Eigen::MatrixXf G;  // Copy of Gram matrix
        float lambda;
        int M;

        SmoothnessCost(Eigen::MatrixXf gram_matrix, float l, int num_centroids)
            : G(std::move(gram_matrix)), lambda(l), M(num_centroids) {
        }

        template <typename T>
        bool operator()(const T* const weights, T* residual) const {
            T smoothness = T(0);
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < M; ++j) {
                    smoothness += T(G(i, j)) * T(weights[i]) * T(weights[j]);
                }
            }

            residual[0] = T(lambda) * smoothness;
            return true;
        }
    };

    class LikelihoodSolver {
    private:
        CodeParameters params_;
        std::vector<BilateralPoint8D> centroids_;
        Eigen::MatrixXf G_;
        Eigen::VectorXf weights_;

    public:
        LikelihoodSolver(const CodeParameters& params) : params_(params) {}

        void setCentroids(const std::vector<BilateralPoint8D>& centroids) {
            centroids_ = centroids;
            buildGramMatrix();
        }

        void buildGramMatrix();

        float rbfKernel(const BilateralPoint8D& p1, const BilateralPoint8D& p2) const;

        void solve(const std::vector<MatchHypothesis>& matches);
        float evaluate(const BilateralPoint8D& point) const;

        const Eigen::VectorXf& getWeights() const { return weights_; }
        const std::vector<BilateralPoint8D>& getCentroids() const { return centroids_; }
    };

} // namespace code

#endif // COHERENCE_REGRESSION_H