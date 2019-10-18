/**
 * @file ICamera.hpp
 * @author Sander Steeghs
 * @date 9 Nov 2017
 * @brief File containing ICamera interface
 *
 * Interface for cameras within the SmartDashCam project
 *
 */

#ifndef BLINDSHOT_ICAMERA_HPP_
#define BLINDSHOT_ICAMERA_HPP_

#include <opencv2/core.hpp> // for cv::Mat
#include <memory> // for std::unique_ptr

/**
 * @brief Interface for cameras within the SmartDashCam project
 */
class ICamera {
public:
    ICamera() { }
    virtual ~ICamera() {}

    /**
     * @brief init Initiate the camera and set correct settings
     */
    virtual void init() = 0;

    /**
     * @brief close Close the camera. call init() again before capture()
     */
    virtual void close() = 0;

    /**
     * @brief method to capture a frame
     * @param output cv::Mat container to store the frame in
     */
    virtual void capture(cv::Mat &output) = 0;

    /**
     * @brief getFrameSize method to request the size of the camera frame
     * @return size of camera frames
     */
    virtual cv::Size getFrameSize() = 0;
};

#endif // BLINDSHOT_ICAMERA_HPP_
