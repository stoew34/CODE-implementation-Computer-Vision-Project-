#pragma once

#include "ceres_config.h"

namespace code {

    // 8D Bilateral point (Eqn 19 in paper)
    struct BilateralPoint8D {
        double x, y;           // Spatial coordinates
        double u, v;           // Motion vector
        double xp, yp;         // Target position (x+u, y+v)
        double o1, o2;         // Feature orientations (simplified affine)

        BilateralPoint8D() : x(0), y(0), u(0), v(0), xp(0), yp(0), o1(0), o2(0) {}

        BilateralPoint8D(double x_, double y_, double u_, double v_,
            double o1_ = 0, double o2_ = 0)
            : x(x_), y(y_), u(u_), v(v_),
            xp(x_ + u_), yp(y_ + v_),
            o1(o1_), o2(o2_) {
        }

        // Convert to Eigen vector
        Eigen::Matrix<double, 8, 1> toVector() const {
            Eigen::Matrix<double, 8, 1> vec;
            vec << x, y, u, v, xp, yp, o1, o2;
            return vec;
        }

        // Squared distance in bilateral space
        double squaredDistance(const BilateralPoint8D& other) const {
            Eigen::Matrix<double, 8, 1> v1 = this->toVector();
            Eigen::Matrix<double, 8, 1> v2 = other.toVector();
            return (v1 - v2).squaredNorm();
        }
    };

    // Feature match with confidence
    struct FeatureMatch {
        cv::KeyPoint kp1, kp2;
        cv::DMatch match;
        double confidence;
        BilateralPoint8D bilateral_point;

        FeatureMatch(const cv::KeyPoint& kp1_, const cv::KeyPoint& kp2_,
            const cv::DMatch& match_, double conf = 0.0)
            : kp1(kp1_), kp2(kp2_), match(match_), confidence(conf) {

            // Convert to bilateral point (Eqn 19)
            bilateral_point = BilateralPoint8D(
                kp1_.pt.x, kp1_.pt.y,
                kp2_.pt.x - kp1_.pt.x,  // u = x2 - x1
                kp2_.pt.y - kp1_.pt.y,  // v = y2 - y1
                kp1_.angle, kp2_.angle  // orientations as affine proxy
            );
        }
    };

    // Regression result
    struct RegressionResult {
        Eigen::VectorXd weights;           // w~ in paper
        Eigen::MatrixXd basis_matrix;      // Basis functions
        double final_cost;
        bool converged;

        RegressionResult() : final_cost(0.0), converged(false) {}
    };

} // namespace code