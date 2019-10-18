#include "blindshot/Cam/DesktopCamera.hpp"

DesktopCamera::DesktopCamera()
{

}

void DesktopCamera::init()
{
    if(!_video.open(0))
    {
        throw std::invalid_argument("Could not open videostream");
    }
}

void DesktopCamera::close()
{
    // Closes automaticaly
}

void DesktopCamera::capture(cv::Mat &output)
{
    cv::Mat frame;
    _video >> output;
}

cv::Size DesktopCamera::getFrameSize()
{
    double width = _video.get(cv::CAP_PROP_FRAME_WIDTH);
    double  height = _video.get(cv::CAP_PROP_FRAME_HEIGHT);
    return cv::Size(width, height);
}
