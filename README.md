# Depedancies
- Opencv
- raspicam (https://thepihut.com/blogs/raspberry-pi-tutorials/16021420-how-to-install-use-the-raspberry-pi-camera)

# Target specification
This application can be defined for two platforms
* desktop
* raspberrypi
Set the flag in ./CMakeLists.txt for the right target and build on the platform

# Building the application
```
mkdir build
cd build
cmake ..
make
```
The application is now build in ./build/src/blindshot{x.x.x}

# Licence
Sander Steeghs
