/*
 * File: fit_ellipse_subpixel.cpp
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
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

static inline cv::Point GetClippedIntegralPoint(const cv::Point2f& pt, int clipX, int clipY);
static inline void UnchainEdgePoint(std::vector<cv::Vec3i>& chain, int idxFrom, int idxTo);
static inline void ChainEdgePoint(std::vector<cv::Vec3i>& chain, int idxFrom, int idxTo);

bool FitEllipseSubPixel(const cv::Mat& gradX, const cv::Mat& gradY, 
    const std::vector<cv::Point>& points, cv::RotatedRect& ellipse)
{
    const int neighbourSize = 5;

    if (points.size() < 6) {
        return false;
    }

    double perim = cv::arcLength(points, true);
    double area = cv::contourArea(points, false);
    double circularity = 4 * M_PI * area / (perim * perim);
    if (circularity < 0.5) {
        return false;
    }

    // Fit an ellipse using given points and find its bounding rect.
    cv::RotatedRect ellipse0 = cv::fitEllipseAMS(points);
    cv::Rect boundRect0 = ellipse0.boundingRect();
    if (boundRect0.width < 8 || boundRect0.height < 8) {
        return false;
    }

    // Inflate bounding rect.
    boundRect0 = cv::Rect(boundRect0.x - neighbourSize, boundRect0.y - neighbourSize,
                boundRect0.width + 2 * neighbourSize, boundRect0.height + 2 * neighbourSize)
        &   cv::Rect(0, 0, gradX.cols, gradX.rows);
    if (boundRect0.area() == 0) {
        return false;
    }

    const int x0 = boundRect0.x, y0 = boundRect0.y;

    // Create mask. Find sub-pixel edge points in 5x5 neighbourhood of each given point.
    cv::Mat mask(boundRect0.size(), CV_8UC1, cv::Scalar::all(0));
    for (const cv::Point& pt : points) {
        mask.at<unsigned char>(
            std::min(boundRect0.height - 1, std::max(0, pt.y - y0)), 
            std::min(boundRect0.width - 1, std::max(0, pt.x - x0))) = 255;
    }
    cv::dilate(mask, mask, cv::Mat::ones(3, 3, CV_8UC1));

    cv::Mat gradXROI(gradX, boundRect0), gradYROI(gradY, boundRect0);

    // Compute lower and upper gradient magnitude thresholds.
    float maxMag = 0.0f;
    for (int r = 0; r != gradXROI.rows; ++r) {
        for (int c = 0; c != gradXROI.cols; ++c) {
            float mag = std::hypot(gradXROI.at<float>(r, c), gradYROI.at<float>(r, c));
            if (mag > maxMag) {
                maxMag = mag;
            }
        }
    }
    float lowerMagThresh = 0.5 * maxMag, upperMagThresh = 0.75 * maxMag;

    // Extract sub-pixel edge points.
    //
    // For detail description on the algorithm, refer to
    //
    // Rafael Grompone von Gioi, Gregory Randall, 
    // A Sub-Pixel Edge Detector: an Implementation of the Canny/Devernay Algorithm, 
    // Image Processing On Line, 7 (2017), pp. 347-372. 
    // https://doi.org/10.5201/ipol.2017.216
    //
    std::vector<cv::Point2f> edgePoints;
    edgePoints.reserve(2 * points.size());
    for (int r = 0; r != gradXROI.rows; ++r) {
        for (int c = 0; c != gradXROI.cols; ++c) {
            if (mask.at<unsigned char>(r, c) == 0) {
                continue;
            }

            float magB = std::hypot(gradXROI.at<float>(r, c), gradYROI.at<float>(r, c));
            if (magB < lowerMagThresh) {
                continue;
            }

            float magA, magC;

            magA = std::hypot(gradXROI.at<float>(r, std::max(0, c - 1)),
                    gradYROI.at<float>(r, std::max(0, c - 1)));
            magC = std::hypot(gradXROI.at<float>(r, std::min(gradXROI.cols - 1, c + 1)),
                    gradYROI.at<float>(r, std::min(gradYROI.cols - 1, c + 1)));
            if (        (magA < magB) 
                    &&  (magB >= magC) 
                    &&  (std::abs(gradXROI.at<float>(r, c))
                        >= std::abs(gradYROI.at<float>(r, c))))
            {
                // Horizontal interpolation
                float lambda = 0.5f * (magA - magC) / (magA - 2.0f * magB + magC + 1.0e-9);
                edgePoints.emplace_back((float)(c) + lambda, (float)(r));
                continue;
            }

            magA = std::hypot(gradXROI.at<float>(std::max(0, r - 1), c),
                    gradYROI.at<float>(std::max(0, r - 1), c));
            magC = std::hypot(gradXROI.at<float>(std::min(gradXROI.rows - 1, r + 1), c),
                    gradYROI.at<float>(std::min(gradYROI.rows - 1, r + 1), c));
            if (        (magA < magB)
                    &&  (magB >= magC)
                    &&  (std::abs(gradXROI.at<float>(r, c))
                        <= std::abs(gradYROI.at<float>(r, c))))
            {
                // Vertical interpolation
                float lambda = 0.5f * (magA - magC) / (magA - 2.0f * magB + magC + 1.0e-9);
                edgePoints.emplace_back((float)(c), (float)(r) + lambda);
                continue;
            }
        }
    }

    // Create edge point index map.
    cv::Mat indexMap(gradXROI.size(), CV_32SC1, cv::Scalar::all(-1));
    for (int i = 0; i != edgePoints.size(); ++i) {
        cv::Point pt = GetClippedIntegralPoint(edgePoints[i], gradXROI.cols, gradXROI.rows);
        int32_t &value = indexMap.at<int32_t>(pt);
        int value0 = value;
        if (value0 == -1) { // not yet assigned.
            value = i;
        }
    }

    // Chain vector: i --> [forward(i), prev(i), valid]
    std::vector<cv::Vec3i> chain(edgePoints.size(), {-1, -1, 0});
    for (int idxP = 0; idxP != edgePoints.size(); ++idxP) {
        const cv::Point2f& ptEdge = edgePoints[idxP];
        const cv::Point pt = GetClippedIntegralPoint(ptEdge, gradXROI.cols, gradXROI.rows);

        // Find best neighbour candidates.
        int idxBestForward = -1, idxBestBackward = -1;
        cv::Vec2f gP(gradXROI.at<float>(pt), gradYROI.at<float>(pt));
        float minDistForward = 1.0e9f, minDistBackward = 1.0e9f;
        for (int r = std::max(0, pt.y - neighbourSize / 2);
                r != std::min(gradXROI.rows - 1, pt.y + neighbourSize / 2); ++r)
        {
            for (int c = std::max(0, pt.x - neighbourSize / 2);
                    c != std::min(gradXROI.cols - 1, pt.x + neighbourSize / 2); ++c)
            {
                if ((r == pt.y) && (c == pt.x)) {
                    continue;
                }

                int idxN = indexMap.at<int32_t>(r, c);
                if (idxN < 0) {
                    continue;
                }

                const cv::Point2f& ptN = edgePoints[idxN];
                cv::Vec2f gN(gradXROI.at<float>(ptN.y, ptN.x), gradYROI.at<float>(ptN.y, ptN.x));
                cv::Vec2f gNOrtho(-gN[1], gN[0]); // rotate gN 90 degrees counter-clockwise.
                cv::Vec2f vecPN(ptN.x - pt.x, ptN.y - pt.y);

                if (gP.dot(gN) <= 0.0f) {
                    continue;
                }

                float dist = std::hypot(vecPN[0], vecPN[1]);
                if (vecPN.dot(gNOrtho) > 0.0f) {
                    if (dist < minDistForward) {
                        idxBestForward = idxN;
                        minDistForward = dist;
                    }
                }
                else {
                    if (dist < minDistBackward) {
                        idxBestBackward = idxN;
                        minDistBackward = dist;
                    }
                }
            }
        }

        if (idxBestForward >= 0) {
            if (chain[idxBestForward][1] < 0) {
                ChainEdgePoint(chain, idxP, idxBestForward);
            }
            else {
                const cv::Point2f& ptBF = edgePoints[idxBestForward];
                float dist1 = std::hypot(ptBF.x - ptEdge.x, ptBF.y - ptEdge.y);
                const cv::Point2f& ptBFPrev = edgePoints[chain[idxBestForward][1]];
                float dist2 = std::hypot(ptBF.x - ptBFPrev.x, ptBF.y - ptBFPrev.y);
                if (dist1 < dist2) {
                    ChainEdgePoint(chain, idxP, idxBestForward);
                }
            }
        }

        if (idxBestBackward >= 0) {
            if (chain[idxBestBackward][0] < 0) {
                ChainEdgePoint(chain, idxBestBackward, idxP);
            }
            else {
                const cv::Point2f& ptBB = edgePoints[idxBestBackward];
                float dist1 = std::hypot(ptBB.x - ptEdge.x, ptBB.y - ptEdge.y);
                const cv::Point2f& ptBBNext = edgePoints[chain[idxBestBackward][0]];
                float dist2 = std::hypot(ptBB.x - ptBBNext.x, ptBB.y - ptBBNext.y);
                if (dist1 < dist2) {
                    ChainEdgePoint(chain, idxBestBackward, idxP);
                }
            }
        }
    }

    // Hysteresis thresholding
    int numValidEdgePoints = 0;
    for (int idx = 0; idx != chain.size(); ++idx) {
        const cv::Point2f& ptEdge = edgePoints[idx];
        const cv::Point pt = GetClippedIntegralPoint(ptEdge, gradXROI.cols, gradXROI.rows);
        float mag = std::hypot(gradXROI.at<float>(pt), gradYROI.at<float>(pt));
        if ((chain[idx][2] == 0) && (mag > upperMagThresh)) {
            chain[idx][2] = 1;
            ++numValidEdgePoints;

            const int idxNext0 = chain[idx][0];
            for (int idxNext = idxNext0; (idxNext >= 0) && (idxNext != idx); 
                    idxNext = chain[idxNext][0]) 
            {
                chain[idxNext][2] = 1;
                ++numValidEdgePoints;
            }

            const int idxPrev0 = chain[idx][1];
            for (int idxPrev = chain[idx][1]; (idxPrev >= 0) && (idxPrev != idx); 
                    idxPrev = chain[idxPrev][1]) 
            {
                chain[idxPrev][2] = 1;
                ++numValidEdgePoints;
            }
        }
    }

    // Collect all valid sub-pixel edge points.
    std::vector<cv::Point2f> subPixelPoints;
    subPixelPoints.reserve(numValidEdgePoints);
    for (int idx = 0; idx != chain.size(); ++idx) {
        if (chain[idx][2] == 1) {
            const cv::Point2f& ptEdge = edgePoints[idx];
            subPixelPoints.emplace_back((float)(x0) + ptEdge.x, (float)(y0) + ptEdge.y);
        }
    }

    if (subPixelPoints.size() < 6) {
        return false;
    }

    ellipse = cv::fitEllipseAMS(subPixelPoints);

    return true;
}

static inline cv::Point GetClippedIntegralPoint(const cv::Point2f& pt, int clipX, int clipY)
{
    int x = pt.x, y = pt.y;

    if (x < 0) {
        x = 0;
    }
    else if (x >= clipX) {
        x = clipX - 1;
    }

    if (y < 0) {
        y = 0;
    }
    else if (y >= clipY) {
        y = clipY - 1;
    }

    return cv::Point(x, y);
}

static inline void UnchainEdgePoint(std::vector<cv::Vec3i>& chain, int idxFrom, int idxTo)
{
    if (idxFrom >= 0) {
        chain[idxFrom][0] = -1;
    }

    if (idxTo >= 0) {
        chain[idxTo][1] = -1;
    }
}

static inline void ChainEdgePoint(std::vector<cv::Vec3i>& chain, int idxFrom, int idxTo)
{
    if (chain[idxFrom][0] >= 0) {
        UnchainEdgePoint(chain, idxFrom, chain[idxFrom][0]);
    }

    if (chain[idxTo][1] >= 0) {
        UnchainEdgePoint(chain, chain[idxTo][1], idxTo);
    }

    // chain FROM-->TO
    chain[idxFrom][0] = idxTo;
    chain[idxTo][1] = idxFrom;
}
