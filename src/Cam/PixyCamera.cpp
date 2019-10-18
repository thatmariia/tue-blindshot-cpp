#include <SmartDashcamDeamon/Cam/PixyCamera.hpp>
#include <pixy.h>
#include <iostream>

using namespace ICT::CoLab::SmartDashCam::Cam;
PixyCamera::PixyCamera()
{

}

PixyCamera::~PixyCamera()
{
    close();
}

void PixyCamera::init()
{
    int ret = pixy_init();
    if(ret != 0)
    {
        pixy_error(ret);
        std::cerr << "Check if:" << std::endl
                  << "\tPixy is properly connected" << std::endl
                  << "\tCheck if udev rules are set" << std::endl << std::endl;
        throw std::runtime_error("Cannot connect to Pixy");
    }

    // Request Pixy firmware version
    uint16_t major; uint16_t minor; uint16_t build;

    ret = pixy_get_firmware_version(&major, &minor, &build);
    if (ret)
    {
        pixy_error(ret);
        throw std::runtime_error("Failed to retrieve Pixy firmware version.");
    }

    std::clog << "Connected to PixyCam " << major << "." << minor << "." << build << std::endl;
}

void PixyCamera::close()
{
    pixy_close();
}

void PixyCamera::detect(std::vector<Detection> &detections)
{
    // Only get new blocks
    if(pixy_blocks_are_new())
    {
        // Retrieve detections from Pixy
        const int bufferlength = 25;
        struct Block blocks[bufferlength];

        int nrBlocks = pixy_get_blocks(bufferlength, &blocks[0]);

        // Copy blocks to detections vector
        for(int i = 0; i < nrBlocks; i++)
        {
            Block block = blocks[i];
            Detection detection(block.signature, block.x, block.y, block.width, block.height);
            detections.push_back(detection);
        }
    }
}
