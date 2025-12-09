#include "glog_fixes.h"
#include "ceres_config.h"
#include "code_pipeline.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char** argv) {
    using namespace std::chrono;

    code::initLogging("CODE Feature Correspondence");

    if (argc != 3) {
        code::logError("Usage: code_project <image1> <image2>");
        code::logError("Example: code_project img1.jpg img2.jpg");
        return -1;
    }

    try {
        auto start_time = high_resolution_clock::now();

        // Load images
        code::logInfo("Loading images...");
        cv::Mat img1_color = cv::imread(argv[1], cv::IMREAD_COLOR);
        cv::Mat img2_color = cv::imread(argv[2], cv::IMREAD_COLOR);

        // Convert to grayscale for SIFT (SIFT uses intensity gradients)
        cv::Mat img1, img2;
        cv::cvtColor(img1_color, img1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2_color, img2, cv::COLOR_BGR2GRAY);

        if (img1.empty() || img2.empty()) {
            code::logError("Cannot read images. Check file paths.");
            return -1;
        }

        code::logInfo("Image 1: " + std::to_string(img1.cols) + "x" + std::to_string(img1.rows));
        code::logInfo("Image 2: " + std::to_string(img2.cols) + "x" + std::to_string(img2.rows));

        // Resize images to similar resolution for better matching
        double scale1 = std::sqrt(img1.cols * img1.rows);
        double scale2 = std::sqrt(img2.cols * img2.rows);
        double max_scale = 2000.0;  // Target image diagonal ~2000 pixels

        if (scale1 > max_scale) {
            double factor = max_scale / scale1;
            cv::resize(img1, img1, cv::Size(), factor, factor, cv::INTER_AREA);
            cv::resize(img1_color, img1_color, cv::Size(), factor, factor, cv::INTER_AREA);
            code::logInfo("Resized image 1 to: " + std::to_string(img1.cols) + "x" + std::to_string(img1.rows));
        }
        if (scale2 > max_scale) {
            double factor = max_scale / scale2;
            cv::resize(img2, img2, cv::Size(), factor, factor, cv::INTER_AREA);
            cv::resize(img2_color, img2_color, cv::Size(), factor, factor, cv::INTER_AREA);
            code::logInfo("Resized image 2 to: " + std::to_string(img2.cols) + "x" + std::to_string(img2.rows));
        }

        // Configure CODE pipeline with corrected normalization:
        // CODE philosophy: Many initial candidates, regression filters bad matches
        code::Config config;
        config.max_features = 8000;

        // Phase 1: Generate MANY initial candidates (permissive Lowe's ratio)
        config.initial_threshold = 0.9f;  // Permissive for viewpoint/rotation variation
        config.final_threshold = 0.9f;    // Also permissive in final step

        // Phase 2: Bilateral clustering - balance between local and global structure
        config.num_clusters = 1000;  // Many clusters for fine-grained local models
        config.spatial_sigma = 1.0;  // Standard for normalized coordinates
        config.descriptor_sigma = 0.5;  // Tighter to group similar motions/orientations

        // Phase 3: Regression and filtering - trust the bilaterally-varying model
        config.likelihood_threshold = 0.55;  // Moderately permissive
        config.spatial_threshold = 1.0;  // Accept all bilaterally-consistent matches
        config.lambda = 0.03;  // Low regularization = better local fitting
        config.huber_epsilon = 0.25;  // Robust to outliers

        code::CodePipeline pipeline(config);

        // Run CODE pipeline
        code::logInfo("Running CODE pipeline...");
        auto matches = pipeline.matchImages(img1, img2);

        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);

        // Display results
        code::logInfo("========================================");
        code::logInfo("RESULTS:");
        code::logInfo("Total matches found: " + std::to_string(matches.size()));
        code::logInfo("Processing time: " + std::to_string(duration.count()) + " ms");
        code::logInfo("========================================");

        // Draw matches if any were found
        if (!matches.empty()) {
            std::vector<cv::KeyPoint> kpts1, kpts2;
            std::vector<cv::DMatch> cv_matches;

            for (size_t i = 0; i < matches.size(); ++i) {
                kpts1.push_back(matches[i].kp1);
                kpts2.push_back(matches[i].kp2);
                // Create new DMatch with correct indices for our vectors
                cv::DMatch dm;
                dm.queryIdx = i;
                dm.trainIdx = i;
                dm.distance = matches[i].match.distance;
                cv_matches.push_back(dm);
            }

            cv::Mat img_matches;
            // Draw matches on color images for better visualization
            cv::drawMatches(img1_color, kpts1, img2_color, kpts2, cv_matches, img_matches,
                cv::Scalar::all(-1), cv::Scalar::all(-1),
                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            // Save result
            cv::imwrite("code_matches_result.jpg", img_matches);
            code::logInfo("Saved matches visualization to: code_matches_result.jpg");

            // Display if possible
            cv::imshow("CODE Matches", img_matches);
            cv::waitKey(3000);  // Show for 3 seconds
            cv::destroyAllWindows();
        }

        // Save match data to file
        std::ofstream match_file("matches.txt");
        if (match_file.is_open()) {
            match_file << "# CODE Feature Matches\n";
            match_file << "# Image1: " << argv[1] << "\n";
            match_file << "# Image2: " << argv[2] << "\n";
            match_file << "# Total matches: " << matches.size() << "\n\n";

            match_file << "idx1, x1, y1, idx2, x2, y2, confidence\n";
            for (size_t i = 0; i < matches.size(); ++i) {
                const auto& match = matches[i];
                match_file << match.match.queryIdx << ", "
                    << match.kp1.pt.x << ", " << match.kp1.pt.y << ", "
                    << match.match.trainIdx << ", "
                    << match.kp2.pt.x << ", " << match.kp2.pt.y << ", "
                    << match.confidence << "\n";
            }
            match_file.close();
            code::logInfo("Saved match data to: matches.txt");
        }

        code::logInfo("=== CODE SUCCESS ===");

    }
    catch (const std::exception& e) {
        code::logError(std::string("Exception: ") + e.what());
        return -1;
    }
    catch (...) {
        code::logError("Unknown exception occurred");
        return -1;
    }

    return 0;
}