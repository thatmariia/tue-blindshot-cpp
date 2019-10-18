/**
 * @file PiCamera.hpp
 * @author Sander Steeghs
 * @date 9 Nov 2017
 * @brief File containing defenition of PiCamera
 *
 * Implements the ICamera interface.
 * This class is a wrapper for the RaspiCamera lib, able to take pictures using the PiCamera
 *
 */

#ifndef BLINDSHOT_PI_CAMERA_HPP_
#define BLINDSHOT_PI_CAMERA_HPP_

#include "blindshot/Cam/ICamera.hpp"
#include <raspicam/raspicam_cv.h>

/**
 * @brief File containing defenition of PiCamera.
 *
 * This class is a wrapper for the RaspiCamera lib, able to take pictures using the PiCamera.
 */
class PiCamera : public ICamera {
public:
    PiCamera();
    virtual ~PiCamera() override;

    /**
     * @brief init Initiate the camera and set correct settings
     */
    virtual void init() override;

    /**
     * @brief close Close the camera, call init() again before capture()
     */
    virtual void close() override;

    /**
     * @brief method to capture a frame from the PiCamera
     * @param output cv::Mat container to store the frame in
     */
    virtual void capture(cv::Mat &output) override;

    /**
     * @brief getFrameSize method to request the size of the camera frame
     * @return size of camera frames
     */
    virtual cv::Size getFrameSize() override;
private:
    // This class can not be copied
    PiCamera(const PiCamera&);
    PiCamera(PiCamera&);

    raspicam::RaspiCam_Cv _cam;
};

#endif // BLINDSHOT_PI_CAMERA_HPP_
