#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "find_calib_pattern.hpp"

static const std::string gs_windowName = "FindHalconCalibBoard()";

void ShowResults(cv::Mat& frame, const std::vector<cv::Point2d>& centers, const cv::Size& patSize)
{
    const int N = patSize.width * patSize.height;

    for (int i = 0; i != N; ++i) {
        const cv::Point2d &pt = centers[i];
        if (i == 0) {
            cv::drawMarker(frame, pt, cv::Scalar(0, 255, 0), cv::MARKER_STAR, 16);
        }
        else if (i == 1) {
            cv::drawMarker(frame, pt, cv::Scalar(0, 255, 0), cv::MARKER_DIAMOND, 16);
        }
        else {
            cv::drawMarker(frame, pt, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 16);
        }

        if (i != N - 1) {
            const cv::Point2d &ptNext = centers[(i + 1) % N];
            cv::line(frame, pt, ptNext, cv::Scalar(255, 0, 0));
        }
    }

    for (int r = 0; r != patSize.height; ++r) {
        std::cout << "Row " << r + 1 << "\n";
        for (int c = 0; c != patSize.width; ++c) {
            std::cout << centers[patSize.width * r + c] << "  ";
            if ((c + 1) % 4 == 0) std::cout << "\n";
        }
        std::cout << "\n";
    }
}

int LiveVideoTest(int argc, char* argv[])
{
    cv::Mat frame;

#ifdef __linux__
    cv::VideoCapture cap("/dev/video0");
#else
    cv::VideoCapture cap(0);
#endif

    std::vector<cv::Point2d> centers;
    const cv::Size patSize(7, 7);

    std::cout << std::fixed << std::setprecision(3);

    for (int nFrame = 1;; ++nFrame) {
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Error: A blank frame was grabbed." << std::endl;
            continue;
        }

        // std::cout << "\x1b[2J\x1b[1;1H" << std::endl;
        std::cout << "Frame #" << nFrame << "\n";

        // if (FindCirclesGridPattern(frame, centers, patSize)) {
        if (FindHalconCalibBoard(frame, centers, patSize)) {
            ShowResults(frame, centers, patSize);
        }
        else {
            for (int thresh = 10; thresh != 250; thresh += 10) {
                // if (FindCirclesGridPattern(frame, centers, patSize, thresh)) {
                if (FindHalconCalibBoard(frame, centers, patSize, thresh)) {
                    ShowResults(frame, centers, patSize);
                    break;
                }
            }
        }

        std::cout << std::endl;

        cv::imshow(gs_windowName, frame);
        cv::waitKey(1);
    }

    return 0;
}

int StillImageTest(int argc, char* argv[])
{
    cv::Mat frame = cv::imread(argv[1]);
    if (frame.empty()) return 1;

    std::vector<cv::Point2d> centers;
    const cv::Size patSize(7, 7);

    std::cout << std::fixed << std::setprecision(3);

    // std::cout << "\x1b[2J\x1b[1;1H" << std::endl;

    if (FindCirclesGridPattern(frame, centers, patSize)) {
    // if (FindHalconCalibBoard(frame, centers, patSize)) {
        ShowResults(frame, centers, patSize);
    }
    else {
        for (int thresh = 10; thresh != 250; thresh += 10) {
            // if (FindCirclesGridPattern(frame, centers, patSize, thresh)) {
            if (FindHalconCalibBoard(frame, centers, patSize, thresh)) {
                ShowResults(frame, centers, patSize);
                break;
            }
        }
    }

    cv::imshow(gs_windowName, frame);
    cv::waitKey(0);

    return 0;
}
    
int main(int argc, char* argv[])
{
    cv::namedWindow(gs_windowName, 
        cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
    cv::resizeWindow(gs_windowName, 800, 600);

    if (argc > 1) {
        return StillImageTest(argc, argv);
    }
    else {
        return LiveVideoTest(argc, argv);
    }
}
