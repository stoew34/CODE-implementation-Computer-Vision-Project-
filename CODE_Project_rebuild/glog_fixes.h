#pragma once

// ============================================
// GLOG Fixes for Windows / Visual Studio
// ============================================

// Prevent Windows macro conflicts
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

// Disable CRT / STL warnings
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

// Disable glog from being included incorrectly
#define CERES_NO_GLOG
#define GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_EXPORT
#define GLOG_NO_EXPORT
#define GLOG_STATIC_DEFINE
#define GOOGLE_GLOG_DLL_DECL

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244)  // Conversion warnings
#pragma warning(disable: 4251)  // DLL interface warnings  
#pragma warning(disable: 4267)  // size_t to int conversion
#pragma warning(disable: 4996)  // Deprecated function warnings
#endif

#ifndef GLOG_USE_GLOG_EXPORT
#define GLOG_USE_GLOG_EXPORT
#endif

#ifndef GLOG_CUSTOM_PREFIX
#define GLOG_CUSTOM_PREFIX
#endif
