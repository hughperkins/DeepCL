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

## linux

### Pre-requisites

- git
- cmake
- gcc
- g++
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU
  - tested using beignet, which provides OpenCL 1.2; and on CUDA 6.5 driver
- opencl-headers
- make 

### Procedure

```bash
git clone --recursive https://github.com/hughperkins/ClConvolve.git
cd ClConvolve
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
- Visual Studio (tested on 2013 Community Edition)
- An OpenCL-compatible driver installed, and OpenCL-compatible GPU

### Procedure

- in git, do `git clone --recursive https://github.com/hughperkins/ClConvolve.git`
- create a subdirectory `build` in the git cloned `ClConvolve` directory
- open cmake, point at the `ClConvolve` directory, and set to build in the `build` subdirectory
  - `configure` then `generate`
- open visual studio, and load any of the projects in the `build` directory
  - change release type to `Release`
  - choose `build` from the `build` menu
- after building, you will need to copy the *.cl files by hand, from the `cl` directory

## Linking

You will need:
- libClConvolve.so (or ClConvolve.dll)
- ~~*.cl files~~ *.cl files no longer needed at runtime :-)  They're packaged inside the dll/so now.

~~The *.cl files should be in the current working directory at the time that you call into any ClConvolve methods.~~


