/*
 * File: find_halcon_calbrd.cpp
 * Author: Neucrede <neucrede@sina.com>
 */

/*
BSD 2-Clause License

Copyright (c) 2018, 
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
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "find_halcon_calbrd.hpp"

extern bool FitEllipseSubPixel(const cv::Mat& gradX, const cv::Mat& gradY, 
    const std::vector<cv::Point>& points, cv::RotatedRect& ellipse); // fit_ellipse_subpixel.cpp

template <typename Tp1, typename Tp2>
static inline float PointLineDistance(const cv::Point_<Tp1>& pt, const cv::Point_<Tp2>& ptLineA,
        const cv::Point_<Tp2>& ptLineB);

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const cv::Point_<Tp2>& pt0);

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const std::vector<int>& indices, const cv::Point_<Tp2>& pt0);

static bool ExtractContours(const cv::Mat& imgGray, int thresh, int total,
        std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point>& outerContour,
        std::vector<cv::Point>& innerContour, std::vector<int>& blobIndicesFiltered);

template <typename Tp>
static bool FindFourCorners(const std::vector<cv::Point_<Tp>>& points, 
        std::vector<int>& cornerIndices);

bool FindHalconCalibBoard(const cv::Mat& img, std::vector<cv::Point2f>& sortedCenterPoints,
        cv::Size patSize, int thresh, bool subPixel)
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
        imgGray = img;
    }

    cv::GaussianBlur(imgGray, imgGray, cv::Size(5, 5), 1.5);

    const size_t total = patSize.width * patSize.height;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> outerContour, innerContour;
    std::vector<int> blobIndicesFiltered;
    if (!ExtractContours(imgGray, thresh, total, contours, outerContour, innerContour,
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
    std::vector<float> minorLengths, majorLengths;
    minorLengths.reserve(total);
    majorLengths.reserve(total);
    float meanMinorLength = 0.0f, meanMajorLength = 0.0f;
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
            ellipse = cv::fitEllipseAMS(contours[idx]);
            if (ellipse.boundingRect().area() == 0) {
                return false;
            }
            else {
                ellipses.push_back(ellipse);
            }
        }

        float minorLength = std::min(ellipse.size.width, ellipse.size.height);
        meanMinorLength += minorLength;

        float majorLength = std::max(ellipse.size.width, ellipse.size.height);
        meanMajorLength += majorLength;
    }

    meanMinorLength /= (float)(total);
    meanMajorLength /= (float)(total);

    float stddevMajorLength = 0.0f, stddevMinorLength = 0.0f;
    for (const cv::RotatedRect& ellipse : ellipses) {
        float minorLength = std::min(ellipse.size.width, ellipse.size.height);
        stddevMinorLength += std::pow(minorLength - meanMinorLength, 2);

        float majorLength = std::max(ellipse.size.width, ellipse.size.height);
        stddevMajorLength += std::pow(majorLength - meanMajorLength, 2);
    }

    stddevMinorLength = std::sqrt(stddevMinorLength / (float)(total));
    stddevMajorLength = std::sqrt(stddevMajorLength / (float)(total));
    
    const float 
        lowMinLen = meanMinorLength - std::max(0.25f * meanMinorLength, 3.0f * stddevMinorLength),
        uppMinLen = meanMinorLength + std::max(0.25f * meanMinorLength, 3.0f * stddevMinorLength),
        lowMajLen = meanMajorLength - std::max(0.25f * meanMajorLength, 3.0f * stddevMajorLength),
        uppMajLen = meanMajorLength + std::max(0.25f * meanMajorLength, 3.0f * stddevMajorLength);

    // * Filter ellipses by their major and minor lengths using 3ss.
    // * Compute center of ellipses.
    std::vector<cv::Point2f> centerPoints;
    centerPoints.reserve(total);
    for (const cv::RotatedRect& ellipse : ellipses) {
        float minorLength = std::min(ellipse.size.width, ellipse.size.height);
        if ((minorLength < lowMinLen) || (minorLength > uppMinLen)) {
            return false;
        }

        float majorLength = std::max(ellipse.size.width, ellipse.size.height);
        if ((majorLength < lowMajLen) || (majorLength > uppMajLen)) {
            return false;
        }

        centerPoints.push_back(ellipse.center);
    }

    // Find out the shortest segment.
    const int M = innerContour.size();
    double minLen = 1.0e9;
    int idxMinLen = 0;
    for (int j = 0; j != M; ++j) {
        const cv::Point &pt = innerContour[j], &ptNext = innerContour[(j + 1) % M];
        double len = std::hypot(pt.x - ptNext.x, pt.y - ptNext.y);
        if (len < minLen) {
            minLen = len;
            idxMinLen = j;
        }
    }

    // Find a point nearest to the shortest segment and set it as the origin.
    const cv::Point &ptLineA = innerContour[idxMinLen],
                    &ptLineB = innerContour[(idxMinLen + 1) % M];
    float minDist = 1.0e9f;
    int idxOrigin = 0;
    for (int i = 0; i != total; ++i) {
        const cv::Point2f& pt = centerPoints[i];
        float dist = PointLineDistance(pt, ptLineA, ptLineB);
        if (dist < minDist) {
            idxOrigin = i;
            minDist = dist;
        }
    }
    const cv::Point2f& ptOrigin = centerPoints[idxOrigin];

    // Find a outer corner point nearest to origin.
    int idxOuter0 = FindNearestPoint(outerContour, ptOrigin);
    if (idxOuter0 < 0) {
        return false;
    }
    const cv::Point& ptOuter0 = outerContour[idxOuter0];

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
    else {
        // Sort corner indices in the order { origin, rear X, rear Y, diagonal }
        std::vector<int> sortedCornerIndices = {
            FindNearestPoint(centerPoints, cornerIndices, ptOrigin),
            FindNearestPoint(centerPoints, cornerIndices, ptOuter1),
            FindNearestPoint(centerPoints, cornerIndices, ptOuter2),
            FindNearestPoint(centerPoints, cornerIndices, outerContour[(idxOuter0 + 2) % 4]) };
        cornerIndices = std::move(sortedCornerIndices);
    }

    // Rectify centre points.
    float stride;
    std::vector<cv::Point2f> rectifiedPoints;
    {
        std::vector<cv::Point2f> srcPoints;
        srcPoints.reserve(4);
        for (int i = 0; i != 4; ++i) {
            srcPoints.push_back(centerPoints[cornerIndices[i]]);
        }

        double len1 = cv::norm(srcPoints[1] - srcPoints[0]),
               len2 = cv::norm(srcPoints[2] - srcPoints[0]);
        stride = (float)(len1 + len2) / (float)(2 * (patSize.width + patSize.height - 2)) + 1.0f;
        std::vector<cv::Point2f> destPoints = {
            {0, 0}, 
            {(float)(patSize.width - 1) * stride, 0},
            {0, (float)(patSize.height - 1) * stride}, 
            {(float)(patSize.width - 1) * stride, (float)(patSize.height - 1) * stride} };

        cv::Mat H = cv::findHomography(srcPoints, destPoints, 0);
        cv::perspectiveTransform(centerPoints, rectifiedPoints, H);
    }

    // Sort centre points in ascending dictionary order.
    sortedCenterPoints.clear();
    sortedCenterPoints.reserve(total);
    for (int r = 0; r != patSize.height; ++r) {
        for (int c = 0; c != patSize.width; ++c) {
            cv::Point2f ptIdeal((float)(c) * stride, (float)(r) * stride);
            int idx = FindNearestPoint(rectifiedPoints, ptIdeal);
            sortedCenterPoints.push_back(centerPoints[idx]);
        }
    }

    return true;
}


template <typename Tp1, typename Tp2>
static inline float PointLineDistance(const cv::Point_<Tp1>& pt, const cv::Point_<Tp2>& ptLineA,
        const cv::Point_<Tp2>& ptLineB)
{
    float xP = pt.x - ptLineA.x, yP = pt.y - ptLineA.y, xV = ptLineB.x - ptLineA.x,
          yV = ptLineB.y - ptLineA.y;
    float vecLen = std::hypot(xV, yV) + 1.0e-9f;
    float crossProd = xP * yV - xV * yP;

    return std::abs(crossProd) / vecLen;
}

template <typename Tp1, typename Tp2>
static int FindNearestPoint(const std::vector<cv::Point_<Tp1>>& points, 
        const cv::Point_<Tp2>& pt0)
{
    cv::Point_<Tp1> pt00 = pt0;

    int idx = -1;
    float minDist = 1.0e9f;
    for (int i = 0; i != points.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[i];
        float dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
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
    float minDist = 1.0e9f;
    for (int i = 0; i != indices.size(); ++i) {
        const cv::Point_<Tp1>& pt = points[indices[i]];
        float dist = std::hypot(pt.x - pt00.x, pt.y - pt00.y);
        if (dist < minDist) {
            idx = indices[i];
            minDist = dist;
        }
    }

    return idx;
}

static bool ExtractContours(const cv::Mat& imgGray, int thresh, int total,
        std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Point>& outerContour,
        std::vector<cv::Point>& innerContour, std::vector<int>& blobIndicesFiltered)
{
    cv::Mat imgMono;
    if (thresh <= 0) {
        cv::threshold(imgGray, imgMono, -1, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    }
    else {
        if (thresh > 250) thresh = 250;
        cv::threshold(imgGray, imgMono, thresh, 255, cv::THRESH_BINARY);
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

        // Simplify approximated contour using a simple mark and sweep procedure.
        //
        // Mark:
        // Walk through 3 adjacent vertices each time and mark the centre one
        // to be deleted if the distance from first to last point is smaller
        // than `distThresh`.
        const int N = approxPointsInner.size();
        float distThresh = cv::arcLength(approxPointsInner, true) / 16.0f;
        std::vector<int> marks(N, 0);
        for (int j = 0; j != N + 1; ++j) {
            const cv::Point &ptFirst = approxPointsInner[j % N], 
                            &ptLast = approxPointsInner[(j + 2) % N];
            float dist = std::hypot(ptFirst.x - ptLast.x, ptFirst.y - ptLast.y);
            if (dist < distThresh) {
                marks[(j + 1) % N] = 1;
            }
        }
        // Sweep:
        innerContour.reserve(N);
        for (int j = 0; j != N; ++j) {
            if (marks[j] != 1) {
                innerContour.push_back(approxPointsInner[j]);
            }
        }

        const int M = innerContour.size();
        if (M < 5) {
            continue;
        }
        else {
            idxInner = i;
            break;
        }
    }

    if (idxInner < 0) {
        return false;
    }

    // Compute blob areas.
    std::vector<int> blobIndices;
    std::vector<float> blobAreas;
    blobIndices.reserve(total);
    blobAreas.reserve(total);
    for (int i = 0; i != numContours; ++i) {
        if (hierarchy[i][3] != idxInner) {
            continue;
        }
        
        const std::vector<cv::Point>& contour = contours[i];
        blobIndices.push_back(i);
        blobAreas.push_back(cv::contourArea(contour));
    }

    // Compute mean and standard deviation of blob areas.
    const int numBlobs = blobIndices.size();
    float meanArea = 0.0f;
    for (float area : blobAreas) {
        meanArea += area;
    }
    meanArea = meanArea / (float)(numBlobs);

    float stddevArea = 0.0f;
    for (float area : blobAreas) {
        stddevArea += std::pow(area - meanArea, 2);
    }
    stddevArea = std::sqrt(stddevArea / (float)(numBlobs));

    // Filter blobs by area using 3-sigma rule of thumb.
    float area1 = meanArea - std::max(0.25f * meanArea, 3.0f * stddevArea), 
          area2 = meanArea + std::max(0.25f * meanArea, 3.0f * stddevArea);
    blobIndicesFiltered.clear();
    blobIndicesFiltered.reserve(numBlobs);
    for (int i = 0; i != numBlobs; ++i) {
        int idx = blobIndices[i];
        float area = blobAreas[i];
        if ((area >= area1) && (area <= area2)) {
            blobIndicesFiltered.push_back(idx);
        }
    }

    return (blobIndicesFiltered.size() == total);
}

template <typename Tp>
static bool FindFourCorners(const std::vector<cv::Point_<Tp>>& points, 
        std::vector<int>& cornerIndices)
{
    std::vector<int> hull;
    cv::convexHull(points, hull);
    if (hull.size() < 4) {
        return false;
    }

    // Compute cosine of angles formed by adjacent 3 vertices.
    std::vector<float> angleCosines;
    angleCosines.reserve(hull.size());
    for (int i = 0; i != hull.size(); ++i) {
        const int K = hull.size();
        cv::Vec2f vec1 = cv::Point2f(points[hull[(i + 1) % K]] - points[hull[i]]);
        cv::Vec2f vec2 = cv::Point2f(points[hull[(i - 1 + K) % K]] - points[hull[i]]);
        float cosAngle = std::abs((float)(vec1.dot(vec2) / (cv::norm(vec1) * cv::norm(vec2))));
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
