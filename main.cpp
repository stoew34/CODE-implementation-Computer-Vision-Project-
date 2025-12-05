#include "code_pipeline.h"
#include "ceres_config.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc, char** argv[]) {
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
        cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            code::logError("Cannot read images. Check file paths.");
            return -1;
        }

        code::logInfo("Image 1: " + std::to_string(img1.cols) + "x" + std::to_string(img1.rows));
        code::logInfo("Image 2: " + std::to_string(img2.cols) + "x" + std::to_string(img2.rows));

        // Configure CODE pipeline
        code::Config config;
        config.max_features = 2000;  // Reduced for testing

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

            for (const auto& match : matches) {
                kpts1.push_back(match.kp1);
                kpts2.push_back(match.kp2);
                cv_matches.push_back(match.match);
            }

            cv::Mat img_matches;
            cv::drawMatches(img1, kpts1, img2, kpts2, cv_matches, img_matches,
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