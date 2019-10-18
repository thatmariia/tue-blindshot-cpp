#include "blindshot/Cam/PiCamera.hpp"

PiCamera::PiCamera()
{

}

PiCamera::~PiCamera()
{

}

void PiCamera::init()
{
    _cam.set(CV_CAP_PROP_FORMAT, CV_8UC3);

    if(!_cam.open() )
    {
        throw std::invalid_argument("Could not open Pi camera");
    }

}

void PiCamera::close()
{
    _cam.release();
}

void PiCamera::capture(cv::Mat &output)
{
    _cam.grab();
    _cam.retrieve(output);
}

cv::Size PiCamera::getFrameSize()
{
    double width = _cam.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = _cam.get(CV_CAP_PROP_FRAME_HEIGHT);
    return cv::Size(width, height);
}
