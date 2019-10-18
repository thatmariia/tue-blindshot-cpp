# Include libraries and source files
set (INCLUDE_DIR_SDW "${INCLUDE_DIR}/blindshot/")
set (SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Find OpenCV package
find_package( OpenCV REQUIRED )

# Find camera libs
if(${TARGET} STREQUAL "raspberrypi")
    set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/include/libraspicam")

    # Find Raspicam package
    find_package( raspicam REQUIRED)

    set(HEADER
            "${HEADER}"
            "${INCLUDE_DIR_SDW}/Cam/PiCamera.hpp"
        )
    set(SOURCE
            "${SOURCE}"
            "Cam/PiCamera.cpp"
        )
else()
    set(HEADER
            "${HEADER}"
            "${INCLUDE_DIR_SDW}/Cam/DesktopCamera.hpp"
        )
    set (SOURCE
        "Cam/DesktopCamera.cpp"
        )
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE
    "${SOURCE}"
    "main.cpp"
    )

set(HEADER
    "${HEADER}"

    "${INCLUDE_DIR_SDW}/Cam/ICamera.hpp"
    "${INCLUDE_DIR_SDW}/color.h"
    )

# Add General Library
add_executable(${LIBRARY_NAME} ${SOURCE} ${HEADER} "${CONFIG_DIR}/blindshot-config.hpp")

# Link OpenCV libs
target_link_libraries( ${LIBRARY_NAME} ${OpenCV_LIBS})

# Link camera libaries
if(${TARGET} STREQUAL "RaspberryPi")
    target_link_libraries( ${LIBRARY_NAME} ${raspicam_CV_LIBS})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")

# Compile with C++11 features
set_target_properties(${LIBRARY_NAME} PROPERTIES COMPILE_FLAGS -std=c++11)
