<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [To build](#to-build)
  - [linux](#linux)
    - [Pre-requisites](#pre-requisites)
    - [Procedure](#procedure)
  - [Windows](#windows)
    - [Pre-requisites](#pre-requisites-1)
    - [Procedure](#procedure-1)
  - [Linking](#linking)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

#To build

Do you want to use DeepCL via python, or from the commandline, or from C++?
* to use from Python, please see [python/README.md](../python/README.md)
* To use from the commandline, or from C++, please continue reading this page :-)

## linux

### Pre-requisites

- git
- cmake
- g++ (should support c++0x; eg 4.4 or better)
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU
  - tested using beignet, which provides OpenCL 1.2; and on CUDA 6.5 driver
- opencl-headers
- make 
- (new) libjpeg62 or compatible, eg `sudo apt-get install libjpeg-turbo8-dev` (libjpeg-turbo is faster than original libjpeg6.2, by around 2-4 times, because it uses SIMD extensions)

### Procedure

```bash
git clone --recursive https://github.com/hughperkins/DeepCL.git
cd DeepCL
mkdir build
cd build
cmake ..
make
```

Note:
* be sure to add `--recursive` when you clone, else when you build it will complain about OpenCLHelper missing (or clew missing)
  * if you do forget, you can experiment with running `git submodule init --recursive`, and then `git submodule update --recursive`
* you might need to play around with commands such as `git submodule update --recursive` occasionally, to pull down new OpenCLHelper updates

## Windows

### Pre-requisites

- git
- cmake
- Visual Studio (current 'standard' build system is: Visual Studio 2010 Express, but should also work on Visual Studio 2008 for Python 2.7, and Visual Studio Express 2013)
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU
- (new) libjpeg62, or compatible, eg [libjpeg-turbo](http://www.libjpeg-turbo.org/Documentation/OfficialBinaries)  (libjpeg-turbo is faster than original libjpeg6.2, by around 2-4 times, because it uses SIMD extensions)
  - if you want, I made a fresh build of libjpeg-turbo 1.4.0:
    - dynamic library (doesnt work for me): [libjpeg-turbo-1.4.0-win32.zip](http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win32.zip) and [libjpeg-turbo-1.4.0-win64.zip](http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win64.zip)
    - static library (works ok for me): [libjpeg-turbo-1.4.0-win32.zip](http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win32-static.zip) and [libjpeg-turbo-1.4.0-win64.zip](http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win64-static.zip)

### Procedure

- in git, do `git clone --recursive https://github.com/hughperkins/DeepCL.git`
- create a subdirectory `build` in the git cloned `DeepCL` directory
- open cmake, point at the `DeepCL` directory, and set to build in the `build` subdirectory
  - `configure` then `generate`
- open visual studio, and load any of the projects in the `build` directory
  - change release type to `Release`
  - choose `build` from the `build` menu

## Linking

If you want to use the DeepCL library from C++, you will need to link with the following libraries:
- libDeepCL.so (or DeepCL.dll, on Windows)
- libOpenCLHelper.so (or OpenCLHelper.dll, on Windows)

