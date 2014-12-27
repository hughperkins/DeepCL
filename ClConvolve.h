// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Timer.h"
#include "OpenCLHelper.h"

class ClConvolve {
public:
    // input: 
    // - numImages * numInputPlanes 2d input images, imageWidth by imageWidth, square
    // - numFilters * numInputPlanes 2d filter images, filterWidth by filterWidth, square
    // - numImages * numFilters 2d result images, imageWidth by imageWidth, square
    // the arrays should be contiguous
    // assumes image is square, and filter is square
    static void convolveImageCubes( int numImages, int numInputPlanes, int numFilters, int imageWidth, int filterWidth,
           int *images, int *filters, int *results ) {
        Timer timer;
        std::cout << "numImageCubes: " << numImages << " numInputPlanes " << numInputPlanes << " numFilters " << numFilters << std::endl;
        OpenCLHelper *cl = new OpenCLHelper(0);
        timer.timeCheck("initialized opencl");

        CLIntWrapper *imagesBuffer = cl->wrap( numImages * numInputPlanes * imageWidth * imageWidth, images );
        CLIntWrapper *filterBuffer = cl->wrap( numFilters * numInputPlanes * filterWidth * filterWidth, filters );
        CLIntWrapper *resultsBuffer = cl->wrap( numImages * numFilters * imageWidth * imageWidth, results );
        imagesBuffer->copyToDevice();
        filterBuffer->copyToDevice();
        timer.timeCheck("copied data to device");

        CLKernel *kernel = 0;
        kernel = cl->buildKernel( "ClConvolve.cl", "convolve_imagecubes_int" );
        kernel->input( 1, &numInputPlanes );
        kernel->input( 1, &numFilters );
        kernel->input( 1, &imageWidth );
        kernel->input( 1, &filterWidth );
        kernel->input( imagesBuffer );
        kernel->input( filterBuffer);
        kernel->output( resultsBuffer );
        int globalSize = numImages * numFilters * imageWidth * imageWidth;
        int workgroupsize = cl->getMaxWorkgroupSize();
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
        kernel->run_1d( globalSize, workgroupsize );
        resultsBuffer->copyToHost();
        timer.timeCheck("run kernel, and copied data back to host");
        delete kernel;  
        delete cl;
        delete imagesBuffer;
        delete filterBuffer;
        delete resultsBuffer;
    }
    // input: 
    // - numImages * numInputPlanes 2d input images, imageWidth by imageWidth, square
    // - numFilters * numInputPlanes 2d filter images, filterWidth by filterWidth, square
    // - numImages * numFilters 2d result images, imageWidth by imageWidth, square
    // the arrays should be contiguous
    // assumes image is square, and filter is square
    static void convolveImageCubes( int numImages, int numInputPlanes, int numFilters, int imageWidth, int filterWidth,
           float *images, float *filters, float *results ) {
        Timer timer;
        std::cout << "numImageCubes: " << numImages << " numInputPlanes " << numInputPlanes << " numFilters " << numFilters << std::endl;
        OpenCLHelper *cl = new OpenCLHelper(0);
        timer.timeCheck("initialized opencl");

        CLFloatWrapper *imagesBuffer = cl->wrap( numImages * numInputPlanes * imageWidth * imageWidth, images );
        CLFloatWrapper *filterBuffer = cl->wrap( numFilters * numInputPlanes * filterWidth * filterWidth, filters );
        CLFloatWrapper *resultsBuffer = cl->wrap( numImages * numFilters * imageWidth * imageWidth, results );
        imagesBuffer->copyToDevice();
        filterBuffer->copyToDevice();
        timer.timeCheck("copied data to device");

        CLKernel *kernel = 0;
        kernel = cl->buildKernel( "ClConvolve.cl", "convolve_imagecubes_float" );
        kernel->input( 1, &numInputPlanes );
        kernel->input( 1, &numFilters );
        kernel->input( 1, &imageWidth );
        kernel->input( 1, &filterWidth );
        kernel->input( imagesBuffer );
        kernel->input( filterBuffer);
        kernel->output( resultsBuffer );
        int globalSize = numImages * numFilters * imageWidth * imageWidth;
        int workgroupsize = cl->getMaxWorkgroupSize();
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
        kernel->run_1d( globalSize, workgroupsize );
        resultsBuffer->copyToHost();
        timer.timeCheck("run kernel, and copied data back to host");
        delete kernel;  
        delete cl;
        delete imagesBuffer;
        delete filterBuffer;
        delete resultsBuffer;
    }
};

