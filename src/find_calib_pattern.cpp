/*
 * File: find_calib_pattern.cpp
 * Author: Neucrede <neucrede@sina.com>
 */

/*
BSD 2-Clause License

Copyright (c) 2018-Now, Neucrede <neucrede@sina.com> 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>
#include <list>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "find_calib_pattern.hpp"

extern bool FitEllipseSubPixel(const cv::Mat& gradX, const cv::Mat& gradY, 
    const std::vector<cv::Point>& points, cv::RotatedRect& ellipse); // fit_ellipse_subpixel.cpp

template <typename Tp1, typename Tp2>
static inline double PointLineDistance(const cv::Point_<Tp1>& pt, const cv::Point_<Tp2>& ptLineA,
        const cv::Point_<Tp2>& ptLineB);

template <typename Tp1, typename Tp2, typename Tp3>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points,
        const cv::Point_<Tp2>& pt0, const cv::Point_<Tp3>& ptLineA, 
        const cv::Point_<Tp3>& ptLineB);

template <typename Tp1, typename Tp2, typename Tp3>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points,
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0, 
        const cv::Point_<Tp3>& ptLineA, const cv::Point_<Tp3>& ptLineB);

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const cv::Point_<Tp2>& pt0);

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0);

template <typename Tp>
static bool FindFourCorners(const std::vector<cv::Point_<Tp>>& points, 
        std::vector<int>& cornerIndices);

template <typename Tp1, typename Tp2>
static bool SortEllipsesAndCenterPoints(cv::Size patSize, const std::vector<cv::Point_<Tp1>>& cornerPoints, 
    const std::vector<cv::RotatedRect>& ellipses, std::vector<cv::RotatedRect>& sortedEllipses,
    const std::vector<cv::Point_<Tp2>>& centerPoints, std::vector<cv::Point_<Tp2>>& sortedCenterPoints);

static bool ExtractContoursHalconCalibBoard(const cv::Mat& imgGray, int thresh, int total,
        std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point>& outerContour,
        std::vector<cv::Point>& innerContour, std::vector<int>& blobIndicesFiltered,
        bool inverseThresh = false);

static bool HierarchicalClustering(const std::vector<cv::Point2d> &points, const cv::Size &patternSz, 
    std::vector<int> &patternPointIndices);

bool FindCirclesGridPattern(const cv::Mat& img, std::vector<cv::Point2d>& sortedCenterPoints,
        cv::Size patSize, int thresh, bool inverseThresh, bool subPixel, const cv::Mat& mask,
        const std::vector<cv::Point2d>& cornerPointsHint, std::vector<cv::RotatedRect>* sortedEllipses)
{
    if (img.empty()) {
        throw std::invalid_argument("img is empty.");
    }

    if ((patSize.width < 2) || (patSize.height < 2)) {
        throw std::invalid_argument("Pattern size can't be smaller than 2x2.");
    }

    // Convert to grayscale image.
    cv::Mat imgGray;
    if (img.channels() != 1) {
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    }
    else {
        imgGray = img.clone();
    }

    // Filter then apply masking.
    cv::GaussianBlur(imgGray, imgGray, cv::Size(5, 5), 1.5);
    if (!mask.empty()) {
        if (mask.type() != CV_8UC1) {
            throw std::invalid_argument("Bad mask data type. Must be CV_8UC1.");
        }

        imgGray = imgGray & mask;
    }

    // Binarization.
    cv::Mat imgMono;
    if (thresh < 0) {
        cv::threshold(imgGray, imgMono, -1, 255, 
            (inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY) + cv::THRESH_OTSU);
    }
    else {
        cv::threshold(imgGray, imgMono, thresh, 255, 
            inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);
    }

    // Extract contours from binarized image.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgMono, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    const size_t total = patSize.width * patSize.height;
    const int numContours = contours.size();
    if (numContours < total) {
        return false;
    }

    // Compute image gradients in horizontal and vertical directions.
    cv::Mat gradX, gradY;
    cv::Sobel(imgGray, gradX, CV_32F, 1, 0);
    cv::Sobel(imgGray, gradY, CV_32F, 0, 1);

    // Fit ellipses blindly.
    std::vector<cv::RotatedRect> ellipses;
    ellipses.reserve(total);
    {
        cv::RotatedRect ellipse;
        for (int idx = 0; idx != numContours; ++idx) {
            cv::Moments moments = cv::moments(contours[idx]);
            double area = moments.m00;
            
            // Filter by area
            {
                const double areaThresh = img.rows * img.cols / patSize.area();
                if (area > areaThresh) {
                    continue;
                }
            }
            
            // Filter by circularity
            {
                double perim = cv::arcLength(contours[idx], true);
                if (4.0 * M_PI * area / (perim * perim + 1.0) < 0.75) {
                    continue;
                }
            }
            
            // Filter by convexity
            {
                std::vector<cv::Point> hull;
                cv::convexHull(contours[idx], hull);
                double hullArea = cv::contourArea(hull);
                if (area / (hullArea + 1.0) < 0.9) {
                    continue;
                }
            }
            
            if (subPixel) {
                if (!FitEllipseSubPixel(gradX, gradY, contours[idx], ellipse)) {
                    continue;
                }
            }
            else if (contours[idx].size() >= 6) {
                ellipse = cv::fitEllipseDirect(contours[idx]);
            }
            else {
                continue;
            }
            
            if (ellipse.boundingRect().area() >= 64) {
                ellipses.push_back(ellipse);
            }
        }
    }

    if (ellipses.size() < total) {
        return false;
    }
    
    // Sweep away overly small blobbs.
    {
        double maxEllipseArea = 0;
        for (const cv::RotatedRect& ellipse : ellipses) {
            double area = M_PI * ellipse.size.area();
            if (area > maxEllipseArea) {
                maxEllipseArea = area;
            }
        }
        
        double area0 = 0.5 * maxEllipseArea;
        
        std::vector<cv::RotatedRect> goodEllipses;
        goodEllipses.reserve(total);
        for (const cv::RotatedRect& ellipse : ellipses) {
            double area = M_PI * ellipse.size.area();
            if (area >= area0) {
                goodEllipses.push_back(ellipse);
            }
        }
        
        ellipses = std::move(goodEllipses);
    }
    
    const int numEllipses = ellipses.size();

    // Find largest cluster of center points.
    std::vector<int> blobIndices;
    {
        std::vector<cv::Point2d> points;
        points.reserve(numEllipses);
        for (const cv::RotatedRect& ellipse : ellipses) {
            points.push_back(ellipse.center);
        }

        if (!HierarchicalClustering(points, patSize, blobIndices)) {
            return false;
        }
    }

    // Compute standard deviations of minor and major lengths of ellipses.
    std::vector<double> minorLengths, majorLengths;
    minorLengths.reserve(total);
    majorLengths.reserve(total);
    double meanMinorLength = 0.0f, meanMajorLength = 0.0f;
    for (int idx : blobIndices) {
        const cv::RotatedRect& ellipse = ellipses[idx];

        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        meanMinorLength += minorLength;

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        meanMajorLength += majorLength;
    }

    meanMinorLength /= (double)(total);
    meanMajorLength /= (double)(total);

    double stddevMajorLength = 0.0f, stddevMinorLength = 0.0f;
    for (int idx : blobIndices) {
        const cv::RotatedRect& ellipse = ellipses[idx];

        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        stddevMinorLength += std::pow(minorLength - meanMinorLength, 2);

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        stddevMajorLength += std::pow(majorLength - meanMajorLength, 2);
    }

    stddevMinorLength = std::sqrt(stddevMinorLength / (double)(total));
    stddevMajorLength = std::sqrt(stddevMajorLength / (double)(total));
    
    // The range of valid major and minor lengths.
    const double 
        lowMinLen = meanMinorLength - std::max(0.5f * meanMinorLength, 3.0f * stddevMinorLength),
        uppMinLen = meanMinorLength + std::max(0.5f * meanMinorLength, 3.0f * stddevMinorLength),
        lowMajLen = meanMajorLength - std::max(0.5f * meanMajorLength, 3.0f * stddevMajorLength),
        uppMajLen = meanMajorLength + std::max(0.5f * meanMajorLength, 3.0f * stddevMajorLength);

    // Filter ellipses by their major and minor lengths using 3ss.
    std::vector<cv::Point2d> centerPoints;
    centerPoints.reserve(total);
    std::vector<cv::RotatedRect> ellipsesFiltered;
    for (int idx : blobIndices) {
        const cv::RotatedRect& ellipse = ellipses[idx];

        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        if ((minorLength < lowMinLen) || (minorLength > uppMinLen)) {
            continue;
        }

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        if ((majorLength < lowMajLen) || (majorLength > uppMajLen)) {
            continue;
        }

        ellipsesFiltered.push_back(ellipse);
        centerPoints.push_back(ellipse.center);
    }

    if (centerPoints.size() != total) {
        return false;
    }

    // If 4 corner points are given in the order { origin, rear X, diagonal, rearY }.
    if (cornerPointsHint.size() == 4) {
        std::vector<cv::RotatedRect> _sortedEllipses;
        if (!SortEllipsesAndCenterPoints(patSize, cornerPointsHint, ellipsesFiltered, _sortedEllipses,
            centerPoints, sortedCenterPoints)) 
        {
            return false;
        }

        if (sortedEllipses) {
            *sortedEllipses = std::move(_sortedEllipses);
        }

        return true;
    }
    // Otherwise...

    // Find four corners among center points.
    std::vector<int> cornerIndices;
    if (!FindFourCorners(centerPoints, cornerIndices)) {
        return false;
    }

    // Sort them in counter-clockwise order.
    {
        std::vector<cv::Point2f> points;
        points.reserve(4);
        for (int i : cornerIndices) {
            points.push_back(cv::Point2f(centerPoints[i]));
        }

        std::vector<int> hull;
        cv::convexHull(points, hull, true, false);

        std::vector<int> cornerIndices2;
        cornerIndices2.reserve(4);
        for (int i : hull) {
            cornerIndices2.push_back(cornerIndices[i]);
        }

        cornerIndices = std::move(cornerIndices2);
    }

    // Find bottom left corner.
    int idxBottomLeft = 0;
    const cv::Point2d ptImageBottomLeft(0.0f, (double)(img.rows));
    double minDist = 1.0e9f;
    for (int i = 0; i != 4; ++i) {
        int idx = cornerIndices[i];
        const cv::Point2d& pt = centerPoints[idx];
        double dist = std::hypot(pt.x - ptImageBottomLeft.x, pt.y - ptImageBottomLeft.y);
        if (dist < minDist) {
            idxBottomLeft = i;
            minDist = dist;
        }
    }

    // Sort in the order { origin = bottom left corner, rear X, diagonal, rearY }.
    std::vector<cv::Point2d> cornerPoints;
    cornerPoints.reserve(4);
    for (int i = idxBottomLeft; i != idxBottomLeft + 4; ++i) {
        cornerPoints.push_back(centerPoints[cornerIndices[i % 4]]);
    }

    std::vector<cv::RotatedRect> _sortedEllipses;
    if (!SortEllipsesAndCenterPoints(patSize, cornerPoints, ellipsesFiltered, _sortedEllipses,
        centerPoints, sortedCenterPoints)) 
    {
        return false;
    }

    if (sortedEllipses) {
        *sortedEllipses = std::move(_sortedEllipses);
    }

    return true;
}

bool FindHalconCalibBoard(const cv::Mat& img, std::vector<cv::Point2d>& sortedCenterPoints,
        cv::Size patSize, int thresh, bool subPixel, std::vector<cv::RotatedRect>* sortedEllipses)
{
    if (img.empty()) {
        throw std::invalid_argument("img is empty.");
    }

    if ((patSize.width < 2) || (patSize.height < 2)) {
        throw std::invalid_argument("Pattern size can't be smaller than 2x2.");
    }

    cv::Mat imgGray;
    if (img.channels() != 1) {
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    }
    else {
        imgGray = img.clone();
    }

    cv::GaussianBlur(imgGray, imgGray, cv::Size(5, 5), 1.5);

    const size_t total = patSize.width * patSize.height;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> outerContour, innerContour;
    std::vector<int> blobIndicesFiltered;
    if (!ExtractContoursHalconCalibBoard(imgGray, thresh, total, contours, outerContour, innerContour,
            blobIndicesFiltered))
    {
        return false;
    }

    // Compute image gradients in horizontal and vertical directions.
    cv::Mat gradX, gradY;
    cv::Sobel(imgGray, gradX, CV_32F, 1, 0);
    cv::Sobel(imgGray, gradY, CV_32F, 0, 1);

    // * Fit ellipses.
    // * Compute means and standard deviations of major and minor axis lengths.
    std::vector<cv::RotatedRect> ellipses;
    ellipses.reserve(total);
    std::vector<double> minorLengths, majorLengths;
    minorLengths.reserve(total);
    majorLengths.reserve(total);
    double meanMinorLength = 0.0f, meanMajorLength = 0.0f;
    cv::RotatedRect ellipse;
    for (int idx : blobIndicesFiltered) {
        if (subPixel) {
            if (FitEllipseSubPixel(gradX, gradY, contours[idx], ellipse)) {
                ellipses.push_back(ellipse);
            }
            else {
                return false;
            }
        }
        else {
            ellipse = cv::fitEllipseDirect(contours[idx]);
            if (ellipse.boundingRect().area() == 0) {
                return false;
            }
            else {
                ellipses.push_back(ellipse);
            }
        }

        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        meanMinorLength += minorLength;

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        meanMajorLength += majorLength;
    }

    meanMinorLength /= (double)(total);
    meanMajorLength /= (double)(total);

    double stddevMajorLength = 0.0f, stddevMinorLength = 0.0f;
    for (const cv::RotatedRect& ellipse : ellipses) {
        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        stddevMinorLength += std::pow(minorLength - meanMinorLength, 2);

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        stddevMajorLength += std::pow(majorLength - meanMajorLength, 2);
    }

    stddevMinorLength = std::sqrt(stddevMinorLength / (double)(total));
    stddevMajorLength = std::sqrt(stddevMajorLength / (double)(total));
    
    const double 
        lowMinLen = meanMinorLength - std::max(0.25f * meanMinorLength, 3.0f * stddevMinorLength),
        uppMinLen = meanMinorLength + std::max(0.25f * meanMinorLength, 3.0f * stddevMinorLength),
        lowMajLen = meanMajorLength - std::max(0.25f * meanMajorLength, 3.0f * stddevMajorLength),
        uppMajLen = meanMajorLength + std::max(0.25f * meanMajorLength, 3.0f * stddevMajorLength);

    // * Filter ellipses by their major and minor lengths using 3ss.
    // * Compute center of ellipses.
    std::vector<cv::Point2d> centerPoints;
    centerPoints.reserve(total);
    std::vector<cv::RotatedRect> ellipsesFiltered;
    for (const cv::RotatedRect& ellipse : ellipses) {
        double minorLength = std::min(ellipse.size.width, ellipse.size.height);
        if ((minorLength < lowMinLen) || (minorLength > uppMinLen)) {
            return false;
        }

        double majorLength = std::max(ellipse.size.width, ellipse.size.height);
        if ((majorLength < lowMajLen) || (majorLength > uppMajLen)) {
            return false;
        }

        ellipsesFiltered.push_back(ellipse);
        centerPoints.push_back(ellipse.center);
    }

    // Map corner points to rect grids.
    std::vector<cv::Point> rectifiedOuterPoints = { 
        {0, 0}, {patSize.width, 0}, {patSize.width, patSize.height}, {0, patSize.width} };
    cv::Mat H = cv::findHomography(outerContour, rectifiedOuterPoints, 0);
    std::vector<cv::Point2d> rectifiedInnerPoints;
    std::vector<cv::Point2d> innerContourDbl;
    innerContourDbl.reserve(innerContour.size());
    for (auto pt : innerContour) {
        innerContourDbl.push_back(cv::Point2d(pt));
    }
    cv::perspectiveTransform(innerContourDbl, rectifiedInnerPoints, H);

    // Find a outer corner nearest to the chamfer.
    int idx0 = 0;
    double maxDist = 0;
    for (int j = 0; j != 4; ++j) {
        const cv::Point& pt = rectifiedOuterPoints[j];
        
        int idx = FindNearestPoint(rectifiedInnerPoints, pt);
        if (idx < 0) {
            return false;
        }

        const cv::Point2d& pt1 = rectifiedInnerPoints[idx];
        double dist = std::hypot(pt.x - pt1.x, pt.y - pt1.y);
        if (dist > maxDist) {
            idx0 = j;
            maxDist = dist;
        }
    }
    const cv::Point& ptOuter0 = outerContour[idx0];
    const int idxOuter0 = idx0;

    // Let the point nearest to the chamfer be the origin.
    idx0 = FindNearestPoint(centerPoints, ptOuter0);
    if (idx0 < 0) {
        return false;
    }
    const cv::Point2d& ptOrigin = centerPoints[idx0];

    // Find 2 outer contour points falls on X and Y axes respectively. #1 --> X, #2 --> Y.
    cv::Point ptOuter1, ptOuter2;
    {
        const cv::Point& ptOuterA = outerContour[(idxOuter0 + 1) % 4];
        const cv::Point& ptOuterB = outerContour[(idxOuter0 + 3) % 4];
        const cv::Vec2i vecA(ptOuterA - ptOuter0), vecB(ptOuterB - ptOuter0);
        int crossProduct = vecA[0] * vecB[1] - vecA[1] * vecB[0];
        if (crossProduct > 0) {
            ptOuter1 = ptOuterB;
            ptOuter2 = ptOuterA;
        }
        else {
            ptOuter1 = ptOuterA;
            ptOuter2 = ptOuterB;
        }
    }

    // Find four corner points.
    std::vector<int> cornerIndices;
    if (!FindFourCorners(centerPoints, cornerIndices)) {
        return false;
    }

    // Sort corner indices in the order { origin, rear X, diagonal, rearY }
    std::vector<int> sortedCornerIndices = {
        FindNearestPoint(centerPoints, cornerIndices, ptOrigin),
        FindNearestPoint(centerPoints, cornerIndices, ptOuter1),
        FindNearestPoint(centerPoints, cornerIndices, outerContour[(idxOuter0 + 2) % 4]),
        FindNearestPoint(centerPoints, cornerIndices, ptOuter2) };
    std::vector<cv::Point2d> cornerPoints;
    cornerPoints.reserve(4);
    for (int i = 0; i != 4; ++i) {
        cornerPoints.push_back(centerPoints[sortedCornerIndices[i]]);
    }

    std::vector<cv::RotatedRect> _sortedEllipses;
    if (!SortEllipsesAndCenterPoints(patSize, cornerPoints, ellipsesFiltered, _sortedEllipses,
        centerPoints, sortedCenterPoints)) 
    {
        return false;
    }

    if (sortedEllipses) {
        *sortedEllipses = std::move(_sortedEllipses);
    }

    return true;
}

template <typename Tp1, typename Tp2>
static inline double PointLineDistance(const cv::Point_<Tp1>& pt, const cv::Point_<Tp2>& ptLineA,
        const cv::Point_<Tp2>& ptLineB)
{
    double xP = pt.x - ptLineA.x, yP = pt.y - ptLineA.y, xV = ptLineB.x - ptLineA.x,
          yV = ptLineB.y - ptLineA.y;
    double vecLen = std::hypot(xV, yV) + 1.0e-9f;
    double crossProd = xP * yV - xV * yP;

    return std::abs(crossProd) / vecLen;
}

template <typename Tp1, typename Tp2, typename Tp3>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points,
        const cv::Point_<Tp2>& pt0, const cv::Point_<Tp3>& ptLineA, 
        const cv::Point_<Tp3>& ptLineB)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    double minDist = 1.0e9f;
    for (int i = 0; i != points.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[i];
        double dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        dist += PointLineDistance(pt, ptLineA, ptLineB);
        if (dist < minDist) {
            idx = i;
            minDist = dist;
        }
    }

    return idx;
}

template <typename Tp1, typename Tp2, typename Tp3>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points,
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0, 
        const cv::Point_<Tp3>& ptLineA, const cv::Point_<Tp3>& ptLineB)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    double minDist = 1.0e9f;
    for (int i = 0; i != points.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[indices[i]];
        double dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        dist += PointLineDistance(pt, ptLineA, ptLineB);
        if (dist < minDist) {
            idx = i;
            minDist = dist;
        }
    }

    return idx;
}

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const cv::Point_<Tp2>& pt0)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    double minDist = 1.0e9f;
    for (int i = 0; i != points.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[i];
        double dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        if (dist < minDist) {
            idx = i;
            minDist = dist;
        }
    }

    return idx;
}

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    double minDist = 1.0e9f;
    for (int i = 0; i != indices.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[indices[i]];
        double dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        if (dist < minDist) {
            idx = indices[i];
            minDist = dist;
        }
    }

    return idx;
}

template <typename Tp>
static bool FindFourCorners(const std::vector<cv::Point_<Tp>>& points, 
        std::vector<int>& cornerIndices)
{
    std::vector<cv::Point2f> points32f;
    points32f.reserve(points.size());
    for (const auto& pt : points) {
        points32f.push_back(cv::Point2f(pt));
    }

    std::vector<int> hull;
    cv::convexHull(points32f, hull);
    if (hull.size() < 4) {
        return false;
    }

    // Compute cosine of angles formed by adjacent 3 vertices.
    std::vector<double> angleCosines;
    angleCosines.reserve(hull.size());
    for (int i = 0; i != hull.size(); ++i) {
        const int K = hull.size();
        cv::Vec2d vec1 = cv::Point2d(points[hull[(i + 1) % K]] - points[hull[i]]);
        cv::Vec2d vec2 = cv::Point2d(points[hull[(i - 1 + K) % K]] - points[hull[i]]);
        double cosAngle = std::abs((double)(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))));
        angleCosines.push_back(cosAngle);
    }

    // Sort angleCosines in ascending order. After sorting, the first 4 vertices 
    // with sharpest angles are considered corners.
    std::vector<int> sortedAngleCosineIndices(angleCosines.size());
    std::iota(sortedAngleCosineIndices.begin(), sortedAngleCosineIndices.end(), 0);
    std::sort(sortedAngleCosineIndices.begin(), sortedAngleCosineIndices.end(), 
            [&angleCosines] (int lhs, int rhs) -> bool {
                return angleCosines[lhs] < angleCosines[rhs];
            }
    );

    cornerIndices.clear();
    cornerIndices.reserve(4);
    for (int i = 0; i != 4; ++i) {
        int idx = hull[sortedAngleCosineIndices[i]];
        cornerIndices.push_back(idx);
    }
    std::sort(cornerIndices.begin(), cornerIndices.end());

    return true;
}

template <typename Tp1, typename Tp2>
static bool SortEllipsesAndCenterPoints(cv::Size patSize, const std::vector<cv::Point_<Tp1>>& cornerPoints, 
    const std::vector<cv::RotatedRect>& ellipses, std::vector<cv::RotatedRect>& sortedEllipses,
    const std::vector<cv::Point_<Tp2>>& centerPoints, std::vector<cv::Point_<Tp2>>& sortedCenterPoints)
{
    // Rectify centre points.
    std::vector<cv::Point2d> rectifiedPoints;
    std::vector<cv::Point2d> srcPoints;
    srcPoints.reserve(4);
    for (int i = 0; i != 4; ++i) {
        srcPoints.push_back(cornerPoints[i]);
    }

    double len1 = cv::norm(srcPoints[1] - srcPoints[0]),
           len2 = cv::norm(srcPoints[2] - srcPoints[0]);
    double stride = (double)(len1 + len2) / (double)(2 * (patSize.width + patSize.height - 2)) + 1.0f;
    std::vector<cv::Point2d> destPoints = {
        {0, 0}, 
        {(double)(patSize.width - 1) * stride, 0},
        {(double)(patSize.width - 1) * stride, (double)(patSize.height - 1) * stride},
        {0, (double)(patSize.height - 1) * stride} };

    cv::Mat H = cv::findHomography(srcPoints, destPoints, 0);
    cv::perspectiveTransform(centerPoints, rectifiedPoints, H);

    // Sort ellipses and centre points in ascending dictionary order.
    sortedCenterPoints.clear();
    sortedCenterPoints.reserve(patSize.width * patSize.height);
    sortedEllipses.clear();
    sortedEllipses.reserve(patSize.width * patSize.height);
    std::vector<int> indices;
    indices.reserve(patSize.width * patSize.height);
    for (int r = 0; r != patSize.height; ++r) {
        for (int c = 0; c != patSize.width; ++c) {
            cv::Point2d ptIdeal((double)(c) * stride, (double)(r) * stride);
            cv::Point2d ptA(0.0, (double)(r) * stride);
            cv::Point2d ptB((double)(patSize.width - 1) * stride, (double)(r) * stride);
            int idx = FindNearestPoint(rectifiedPoints, ptIdeal, ptA, ptB);
            
            sortedCenterPoints.push_back(centerPoints[idx]);
            sortedEllipses.push_back(ellipses[idx]);
            indices.push_back(idx);
        }
    }

    int N = patSize.width;
    cv::Point2d pt0 = rectifiedPoints[indices[0]], pt1 = rectifiedPoints[indices[1]],
                ptN = rectifiedPoints[indices[N - 1]];
    double dist01 = std::hypot(pt0.x - pt1.x, pt0.y - pt1.y);
    double dist0N = std::hypot(pt0.x - ptN.x, pt0.y - ptN.y);

    return ((dist0N > (double)(N - 2) * dist01) && (dist0N < 1.2 * (double)(N - 1) * dist01));
}

static bool HierarchicalClustering(const std::vector<cv::Point2d> &points, const cv::Size &patternSz, 
    std::vector<int> &patternPointIndices)
{
    // This function was ported from OpenCV 3.4.5 calib3d module with modification.
    //
    // sources/modules/calib3d/src/circlesgrid.cpp  
    // lines 70 - 138
    //

/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

    int j, n = (int)points.size();
    size_t pn = static_cast<size_t>(patternSz.area());

    patternPointIndices.clear();
    patternPointIndices.reserve(pn);

    if (pn >= points.size())
    {
        if (pn == points.size()) {
            patternPointIndices.resize(pn);
            std::iota(patternPointIndices.begin(), patternPointIndices.end(), 0);
            return true;
        }
        else {
            return false;
        }
    }

    cv::Mat dists(n, n, CV_64FC1, cv::Scalar(0));
    cv::Mat distsMask(dists.size(), CV_8UC1, cv::Scalar(0));
    for(int i = 0; i < n; i++)
    {
        for(j = i+1; j < n; j++)
        {
            dists.at<double>(i, j) = (double)cv::norm(points[i] - points[j]);
            distsMask.at<uchar>(i, j) = 255;
            distsMask.at<uchar>(j, i) = 255;//distsMask.at<uchar>(i, j);
            dists.at<double>(j, i) = dists.at<double>(i, j);
        }
    }

    std::vector<std::list<size_t> > clusters(points.size());
    for(size_t i=0; i<points.size(); i++)
    {
        clusters[i].push_back(i);
    }

    int patternClusterIdx = 0;
    while(clusters[patternClusterIdx].size() < pn)
    {
        cv::Point minLoc;
        cv::minMaxLoc(dists, 0, 0, &minLoc, 0, distsMask);
        int minIdx = std::min(minLoc.x, minLoc.y);
        int maxIdx = std::max(minLoc.x, minLoc.y);

        distsMask.row(maxIdx).setTo(0);
        distsMask.col(maxIdx).setTo(0);
        cv::Mat tmpRow = dists.row(minIdx);
        cv::Mat tmpCol = dists.col(minIdx);
        cv::min(dists.row(minLoc.x), dists.row(minLoc.y), tmpRow);
        tmpRow = tmpRow.t();
        tmpRow.copyTo(tmpCol);

        clusters[minIdx].splice(clusters[minIdx].end(), clusters[maxIdx]);
        patternClusterIdx = minIdx;
    }

    if(clusters[patternClusterIdx].size() < static_cast<size_t>(patternSz.area()))
    {
        return false;
    }

    for(std::list<size_t>::iterator it = clusters[patternClusterIdx].begin(); it != clusters[patternClusterIdx].end();++it)
    {
        patternPointIndices.push_back(*it);
    }

    return true;
}

static bool ExtractContoursHalconCalibBoard(const cv::Mat& imgGray, int thresh, int total,
        std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point>& outerContour,
        std::vector<cv::Point>& innerContour, std::vector<int>& blobIndicesFiltered,
        bool inverseThresh)
{
    cv::Mat imgMono;
    if (thresh < 0) {
        cv::threshold(imgGray, imgMono, -1, 255, 
            (inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY) + cv::THRESH_OTSU);
    }
    else {
        cv::threshold(imgGray, imgMono, thresh, 255, 
            inverseThresh ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);
    }

    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgMono, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    const int numContours = contours.size();

    // Find inner and outer contours.
    int idxInner = -1;
    for (int i = 0; i != numContours; ++i) {
        // Count number of child contours.
        int n = 0;
        for (int j = hierarchy[i][2]; j >= 0; j = hierarchy[j][0], ++n);

        // Inner contour should contain `total` circles and must have
        // a valid parent contour.
        if ((n < total) || (n > total + 5) || (hierarchy[i][3] < 0)) {
            continue;
        }

        // Try approximate parent contour.
        // The outer contour must be a quadrilateral.
        std::vector<cv::Point> outerContourApprox;
        double eps = cv::arcLength(contours[hierarchy[i][3]], true) / 32.0;
        cv::approxPolyDP(contours[hierarchy[i][3]], outerContourApprox, eps, true);

        std::vector<int> outerCornerIndices;
        if (!FindFourCorners(outerContourApprox, outerCornerIndices)) {
            continue;
        }
        else {
            outerContour.clear();
            outerContour.reserve(4);
            for (int idx : outerCornerIndices) {
                outerContour.push_back(outerContourApprox[idx]);
            }
        }

        // Try approximate current contour.
        std::vector<cv::Point> approxPointsInner;
        cv::approxPolyDP(contours[i], approxPointsInner, 3, true);

        // The inner border of a Halcon calibration board pattern is a rectangle
        // with one of its corner chamfered hence ideal total number of vertices is 5.
        if (approxPointsInner.size() < 5) {
            continue;
        }
        else {
            idxInner = i;
            innerContour = std::move(approxPointsInner);
            break;
        }
    }

    if (idxInner < 0) {
        return false;
    }

    // Compute blob areas.
    std::vector<int> blobIndices;
    std::vector<double> blobAreas;
    blobIndices.reserve(total);
    blobAreas.reserve(total);
    for (int i = 0; i != numContours; ++i) {
        if (hierarchy[i][3] != idxInner) {
            continue;
        }
        
        const std::vector<cv::Point>& contour = contours[i];
        
        cv::Moments moments = cv::moments(contour);
        double area = moments.m00;
        
        // Filter by area
        {
            const double areaThresh = imgGray.rows * imgGray.cols / total;
            if (area > areaThresh) {
                continue;
            }
        }
            
        // Filter by circularity
        {
            double perim = cv::arcLength(contour, true);
            if (4.0 * M_PI * area / (perim * perim + 1.0) < 0.75) {
                continue;
            }
        }
        
        // Filter by convexity
        {
            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            double hullArea = cv::contourArea(hull);
            if (area / (hullArea + 1.0) < 0.9) {
                continue;
            }
        }
        
        if (contour.size() >= 9) {
            blobIndices.push_back(i);
            blobAreas.push_back(cv::contourArea(contour));
        }
    }

    // Compute mean and standard deviation of blob areas.
    const int numBlobs = blobIndices.size();
    double meanArea = 0.0f;
    for (double area : blobAreas) {
        meanArea += area;
    }
    meanArea = meanArea / (double)(numBlobs);

    double stddevArea = 0.0f;
    for (double area : blobAreas) {
        stddevArea += std::pow(area - meanArea, 2);
    }
    stddevArea = std::sqrt(stddevArea / (double)(numBlobs));

    // Filter blobs by area using 3-sigma rule of thumb.
    double area1 = meanArea - std::max(0.25f * meanArea, 3.0f * stddevArea), 
          area2 = meanArea + std::max(0.25f * meanArea, 3.0f * stddevArea);
    blobIndicesFiltered.clear();
    blobIndicesFiltered.reserve(numBlobs);
    for (int i = 0; i != numBlobs; ++i) {
        int idx = blobIndices[i];
        double area = blobAreas[i];
        if ((area >= area1) && (area <= area2)) {
            blobIndicesFiltered.push_back(idx);
        }
    }

    return (blobIndicesFiltered.size() == total);
}

