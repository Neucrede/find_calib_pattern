/**
 * \file find_calib_pattern.hpp
 *
 * \author Neucrede neucrede@sina.com
 * \version 1.1
 *
 * \brief Subroutines for the search of calibration patterns.
 *
 */

#ifndef __FIND_CALIB_PATTERN_HPP__
#define __FIND_CALIB_PATTERN_HPP__

#include <vector>
#include <opencv2/core.hpp>

/*
namespace DeviceCalibration
{
*/

bool FindCirclesGridPattern(const cv::Mat& img, std::vector<cv::Point2d>& sortedCenterPoints,
        cv::Size patSize = cv::Size(7, 7), int thresh = -1, bool inverseThresh = false,
        bool subPixel = true, const cv::Mat& mask = cv::Mat(),
        const std::vector<cv::Point2d>& cornerPointsHint = std::vector<cv::Point2d>(),
        std::vector<cv::RotatedRect>* sortedEllipses = nullptr);

bool FindHalconCalibBoard(const cv::Mat& img, std::vector<cv::Point2d>& sortedCenterPoints,
        cv::Size patSize = cv::Size(7, 7), int thresh = -1, bool subPixel = true,
        std::vector<cv::RotatedRect>* sortedEllipses = nullptr);

/*
}
*/


#endif /* __FIND_CALIB_PATTERN_HPP__ */

