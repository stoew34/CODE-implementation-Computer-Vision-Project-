#include "bilateral_clusterer.h"
#include <random>
#include <limits>

namespace code {

    BilateralClusterer::BilateralClusterer(const Config& config) : config_(config) {}

    std::vector<BilateralPoint8D> BilateralClusterer::cluster(
        const std::vector<BilateralPoint8D>& points) const {

        if (points.empty() || config_.num_clusters >= static_cast<int>(points.size())) {
            return points;
        }

        std::vector<BilateralPoint8D> centroids;
        std::vector<int> assignments(points.size(), -1);

        // Initialize centroids with k-means++
        initializeCentroids(points, centroids);

        // Simple k-means iteration
        for (int iter = 0; iter < 10; ++iter) {
            // Assign points to nearest centroids
            assignToCentroids(points, centroids, assignments);

            // Update centroids - compute mean of each cluster
            std::vector<BilateralPoint8D> sum_centroids(config_.num_clusters);
            std::vector<int> counts(config_.num_clusters, 0);

            // Initialize sums to zero
            for (int i = 0; i < config_.num_clusters; ++i) {
                sum_centroids[i] = BilateralPoint8D();
            }

            // Accumulate sums
            for (size_t i = 0; i < points.size(); ++i) {
                int cluster = assignments[i];
                if (cluster >= 0 && cluster < config_.num_clusters) {
                    const auto& p = points[i];
                    sum_centroids[cluster].x += p.x;
                    sum_centroids[cluster].y += p.y;
                    sum_centroids[cluster].u += p.u;
                    sum_centroids[cluster].v += p.v;
                    sum_centroids[cluster].xp += p.xp;
                    sum_centroids[cluster].yp += p.yp;
                    sum_centroids[cluster].o1 += p.o1;
                    sum_centroids[cluster].o2 += p.o2;
                    counts[cluster]++;
                }
            }

            // Compute averages and keep centroids with points
            centroids.clear();
            for (int i = 0; i < config_.num_clusters; ++i) {
                if (counts[i] > 0) {
                    BilateralPoint8D centroid;
                    centroid.x = sum_centroids[i].x / counts[i];
                    centroid.y = sum_centroids[i].y / counts[i];
                    centroid.u = sum_centroids[i].u / counts[i];
                    centroid.v = sum_centroids[i].v / counts[i];
                    centroid.xp = sum_centroids[i].xp / counts[i];
                    centroid.yp = sum_centroids[i].yp / counts[i];
                    centroid.o1 = sum_centroids[i].o1 / counts[i];
                    centroid.o2 = sum_centroids[i].o2 / counts[i];
                    centroids.push_back(centroid);
                }
            }
        }

        return centroids;
    }

    Eigen::MatrixXd BilateralClusterer::computeKernelMatrix(
        const std::vector<BilateralPoint8D>& centroids) const {

        int m = static_cast<int>(centroids.size());
        Eigen::MatrixXd kernel_matrix(m, m);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                kernel_matrix(i, j) = gaussianKernel(centroids[i], centroids[j]);
            }
        }

        return kernel_matrix;
    }

    Eigen::MatrixXd BilateralClusterer::computeBasisMatrix(
        const std::vector<BilateralPoint8D>& points,
        const std::vector<BilateralPoint8D>& centroids) const {

        int n = static_cast<int>(points.size());
        int m = static_cast<int>(centroids.size());
        Eigen::MatrixXd basis_matrix(n, m);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                basis_matrix(i, j) = gaussianKernel(points[i], centroids[j]);
            }
        }

        return basis_matrix;
    }

    Eigen::VectorXd BilateralClusterer::computeBasisForPoint(
        const BilateralPoint8D& point,
        const std::vector<BilateralPoint8D>& centroids) const {

        int m = static_cast<int>(centroids.size());
        Eigen::VectorXd basis(m);

        for (int j = 0; j < m; ++j) {
            basis(j) = gaussianKernel(point, centroids[j]);
        }

        return basis;
    }

    double BilateralClusterer::gaussianKernel(const BilateralPoint8D& p1,
        const BilateralPoint8D& p2) const {
        // Bilateral kernel: separate spatial and descriptor components (Eqn 11 in paper)
        // Spatial: (x, y)
        double spatial_dist_sq = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);

        // Descriptor: (u, v, xp, yp, o1, o2) - motion and appearance
        double desc_dist_sq = (p1.u - p2.u) * (p1.u - p2.u) +
                              (p1.v - p2.v) * (p1.v - p2.v) +
                              (p1.xp - p2.xp) * (p1.xp - p2.xp) +
                              (p1.yp - p2.yp) * (p1.yp - p2.yp) +
                              (p1.o1 - p2.o1) * (p1.o1 - p2.o1) +
                              (p1.o2 - p2.o2) * (p1.o2 - p2.o2);

        double spatial_sigma_sq = config_.spatial_sigma * config_.spatial_sigma;
        double desc_sigma_sq = config_.descriptor_sigma * config_.descriptor_sigma;

        return std::exp(-spatial_dist_sq / (2.0 * spatial_sigma_sq) - desc_dist_sq / (2.0 * desc_sigma_sq));
    }

    void BilateralClusterer::initializeCentroids(const std::vector<BilateralPoint8D>& points,
        std::vector<BilateralPoint8D>& centroids) const {

        int k = config_.num_clusters;
        if (k <= 0 || points.empty()) return;

        centroids.clear();
        std::vector<double> min_distances(points.size(),
            std::numeric_limits<double>::max());

        // First centroid: random point
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, static_cast<int>(points.size()) - 1);

        centroids.push_back(points[dis(gen)]);

        // K-means++ initialization
        for (int i = 1; i < k; ++i) {
            // Update distances to nearest centroid
            double total_distance = 0.0;
            for (size_t j = 0; j < points.size(); ++j) {
                double dist = points[j].squaredDistance(centroids.back());
                if (dist < min_distances[j]) {
                    min_distances[j] = dist;
                }
                total_distance += min_distances[j];
            }

            // Choose next centroid with probability proportional to distance^2
            std::uniform_real_distribution<> prob_dist(0.0, total_distance);
            double random_value = prob_dist(gen);

            double cumulative = 0.0;
            for (size_t j = 0; j < points.size(); ++j) {
                cumulative += min_distances[j];
                if (cumulative >= random_value) {
                    centroids.push_back(points[j]);
                    break;
                }
            }
        }
    }

    void BilateralClusterer::assignToCentroids(const std::vector<BilateralPoint8D>& points,
        const std::vector<BilateralPoint8D>& centroids,
        std::vector<int>& assignments) const {

        for (size_t i = 0; i < points.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_cluster = -1;

            for (size_t j = 0; j < centroids.size(); ++j) {
                double dist = points[i].squaredDistance(centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = static_cast<int>(j);
                }
            }

            assignments[i] = best_cluster;
        }
    }

} // namespace code