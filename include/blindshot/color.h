#ifndef COLOR_HPP_
#define COLOR_HPP_

#include <opencv2/opencv.hpp>
/**
 * @brief Interface for the SmartDashCam project
 */

class Color {
public:
    cv::Scalar high;
    cv::Scalar low;
    std::string name;

    Color(std::string name, cv::Scalar low, cv::Scalar high) :
        high(high),
        low(low),
        name(name) {

    }
};

#endif // COLOR_HPP_
