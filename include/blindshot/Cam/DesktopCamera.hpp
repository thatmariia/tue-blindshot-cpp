#ifndef DESKTOP_CAMERA_HPP_
#define DESKTOP_CAMERA_HPP_

#include "blindshot/Cam/ICamera.hpp"

#include <opencv2/highgui.hpp>  // for cv::VideoCapture
#include <chrono>               // for time

class DesktopCamera : public ICamera {
public:
    /**
     * @brief PlaybackCamera
     */
    DesktopCamera();
    virtual ~DesktopCamera() override {}

    /**
     * @brief init Initiate the video stream and set correct settings
     */
    virtual void init() override;

    /**
     * @brief close Close the video file
     */
    virtual void close() override;

    /**
     * @brief method to capture a frame from the file
     * @param output method to capture the next frame from the file.
     */
    virtual void capture(cv::Mat &output) override;

    /**
     * @brief getFrameSize method to request the size of the camera frame
     * @return size of camera frames
     */
    virtual cv::Size getFrameSize() override;
private:
    // This class can not be copied
    DesktopCamera(const DesktopCamera&);
    DesktopCamera(DesktopCamera&);

    cv::VideoCapture _video;
};

#endif // SMART_DASH_PLAYBACK_CAMERA_HPP_
