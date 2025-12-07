#pragma once

namespace code {

    struct Config {
        // Feature detection (A-SIFT)
        int max_features = 2000;           // Balanced for speed and coverage
        float contrast_threshold = 0.02f;  // Even lower for more candidates
        float edge_threshold = 15.0f;      // Higher to reject edge-like features
        float sigma = 1.6f;

        // A-SIFT simulation parameters
        bool use_asift_simulation = true;
        int num_tilt_angles = 5;          // Number of affine simulation angles
        double tilt_range = 70.0;          // Max tilt angle in degrees

        // Matching thresholds (Fig 6)
        float initial_threshold = 0.75f;  // More permissive to find more candidates
        float final_threshold = 0.80f;    // Balanced for quality matches     

        // Bilateral clustering (Sec 2.2)
        int num_clusters = 150;           // More clusters for finer detail (keeps wall suppression)
        double spatial_sigma = 1.5;       // Tighter spatial grouping (normalized space)
        double descriptor_sigma = 0.4;    // Tighter descriptor grouping

        // Coherence regression (Sec 3)
        double lambda = 0.5;              // Less regularization for more flexibility
        double huber_epsilon = 0.5;       // Larger for robustness to outliers
        int max_iterations = 50;          // Faster convergence (reduced from 150)
        double function_tolerance = 1e-6; // Slightly relaxed for speed

        // Decision boundaries (Sec 3)
        double likelihood_threshold = 0.50;  // More permissive for more candidates
        double spatial_threshold = 1.0;      // Balanced threshold     

        // Ceres solver
        int num_threads = 8;              // Use more threads for parallelization
        bool use_nonmonotonic_steps = true;
    };

} // namespace code