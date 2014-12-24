ClConvolve
==========

Current status: DRAFT, IN PROGRESS

OpenCL library to run convolutions on stacks of input images, using stacks of filters

API
===

Simply call:

    ClConvolve::convolveImageCubes( int numImages, int numInputPlanes, int numFilters, int imageWidth, int filterWidth,
           int *images, int *filters, int *results );

- you need to provide the images, filters, and results arrays as contiguous arrays of integers
- the images array should consist of numImages cubes of images
  - each cube should have numInputPlanes planes
    - each plane should have imageWidth * imageWidth values, representing one plane of the input image
- the filters array should consist of numFilters * numInputPlanes images, each of filterWidth * filterWidth ints
- the output results array should be able to hold numImages * numFilters * imageWidth * imageWidth ints

Pre-requisites
==============

- git
- cmake
- gcc
- g++
- opencl-headers
- make 

To build
========

    git clone --recursive https://github.com/hughperkins/ClConvolve.git
    cd ClConvolve
    mkdir build
    cd build
    cmake ..
    make ClConvolve

