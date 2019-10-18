cmake_minimum_required(VERSION 2.8)
project(blindshot)

# Set the target platform. Uncomment the preferred target.
set(TARGET "desktop")
#set(TARGET "raspberrypi")

set(VERSION_MAJOR "0")
set(VERSION_MINOR "1")
set(VERSION_PATCH "0")
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

set (LIBRARY_NAME "${PROJECT_NAME}.${VERSION}")

set (INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set (CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/platform/${TARGET}")

include_directories(${INCLUDE_DIR} ${CONFIG_DIR})

# include common directories
add_subdirectory(src)
