#pragma once

namespace code {

    struct Config {
        // Feature detection (A-SIFT)
        int max_features = 2000;
        float contrast_threshold = 0.04f;
        float edge_threshold = 10.0f;
        float sigma = 1.6f;

        // Matching thresholds (Fig 6)
        float initial_threshold = 0.86f;  
        float final_threshold = 1.0f;     

        // Bilateral clustering (Sec 2.2)
        int num_clusters = 100;           // M in paper
        double spatial_sigma = 10.0;      
        double descriptor_sigma = 0.5;

        // Coherence regression (Sec 3)
        double lambda = 1.0;              // Smoothness weight 
        double huber_epsilon = 0.1;       
        int max_iterations = 100;
        double function_tolerance = 1e-6;

        // Decision boundaries (Sec 3)
        double likelihood_threshold = 0.6;   
        double spatial_threshold = 0.01;     

        // Ceres solver
        int num_threads = 4;
        bool use_nonmonotonic_steps = true;
    };

} // namespace code