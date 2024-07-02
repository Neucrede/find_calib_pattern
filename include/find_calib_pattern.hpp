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

bool FindCirclesGridPattern(const cv::Mat& img, std::vector<cv::Point2f>& sortedCenterPoints,
        cv::Size patSize = cv::Size(7, 7), int thresh = -1, bool inverseThresh = false,
        bool subPixel = true, const cv::Mat& mask = cv::Mat(),
        const std::vector<cv::Point2f>& cornerPointsHint = std::vector<cv::Point2f>());

/**
 * \brief Find HALCON calibration board pattern from given image and return centers
 * in the grid of solid circle pattern if all of them were found.
 *
 * \param [in]      img 
 *                  An 1-channel grayscale image (CV_8UC1) or a 3-channel BGR image (CV_8UC3).
 * \param [out]     sortedCenterPoints 
 *                  Output array of center points sorted row by row.
 * \param [in]      patSize
 *                  Number of circles per row and column.
 *                  (patSize = cv::Size(points_per_row, points_per_column).
 *                  Defaulted to (7, 7) which is the size of the off the shelf HALCON
 *                  calibration plates.
 * \param [in]      thresh
 *                  Threshold value used for image binarization. Setting this value to a
 *                  negative number indicates that the threshold is internally determined
 *                  by means of an adaptive method, currently Otsu. This is the default 
 *                  case.
 * \param [in]      subPixel
 *                  Indicates whether sub-pixel edge extraction is used to find the edges
 *                  of elliptical blobs. Default value is true.
 *
 * \return This function returns true if all circles were found, otherwise it returns
 * false.
 */
bool FindHalconCalibBoard(const cv::Mat& img, std::vector<cv::Point2f>& sortedCenterPoints,
        cv::Size patSize = cv::Size(7, 7), int thresh = -1, bool subPixel = true);


#endif /* __FIND_CALIB_PATTERN_HPP__ */

