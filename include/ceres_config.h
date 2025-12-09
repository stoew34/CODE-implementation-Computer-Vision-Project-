#pragma once

#include "glog_fixes.h"
// Standard includes
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <random>

// Eigen BEFORE Ceres
#include <Eigen/Dense>
#include <Eigen/Core>

// Ceres WITHOUT GLOG
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// Simple logging that won't conflict
namespace code {
    inline void initLogging(const char* name) {
        std::cout << "[CODE] Initializing: " << name << std::endl;
    }

    inline void logInfo(const std::string& msg) {
        std::cout << "[INFO] " << msg << std::endl;
    }

    inline void logError(const std::string& msg) {
        std::cerr << "[ERROR] " << msg << std::endl;
    }

    inline void logWarning(const std::string& msg) {
        std::cout << "[WARNING] " << msg << std::endl;
    }
}