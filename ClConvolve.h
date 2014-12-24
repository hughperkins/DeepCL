// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "OpenCLHelper.h"

class ClConvolve {
public:
    // input: 
    // - one single image
    // - one filter
    // the array should be contiguous
    // assumes image is square, and filter is square
    static void convolveImage( int imageWidth, int filterWidth, int *image, int *filter, int *result ) {
        OpenCLHelper *cl = new OpenCLHelper(0);
        CLIntWrapper *imagesBuffer = cl->intWrapper( imageWidth * imageWidth, image );
        CLIntWrapper *filterBuffer = cl->intWrapper( filterWidth * filterWidth, filter );
        CLIntWrapper *resultsBuffer = cl->intWrapper( imageWidth * imageWidth, result );
        imagesBuffer->copyToDevice();
        filterBuffer->copyToDevice();

        CLKernel *kernel = 0;
        kernel = cl->   buildKernel( "../ClConvolve.cl", "convolve_ints" );
        kernel->input( 1, &imageWidth );
        kernel->input( 1, &filterWidth );
        kernel->input( imagesBuffer );
        kernel->input( filterBuffer);
        kernel->output( resultsBuffer );
        int globalSize = imageWidth * imageWidth;
        int workgroupsize = cl->getMaxWorkgroupSize();
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
        kernel->run_1d( globalSize, workgroupsize );
        resultsBuffer->copyToHost();
        delete kernel;
        delete cl;
        delete imagesBuffer;
        delete filterBuffer;
        delete resultsBuffer;
    }

    // input: 
    // - N images
    // - one filter
    // the arrays should be contiguous
    // assumes image is square, and filter is square
    static void convolveImages( int N, int imageWidth, int filterWidth, int *images, int *filter, int *results ) {
        OpenCLHelper *cl = new OpenCLHelper(0);

        CLIntWrapper *imagesBuffer = cl->intWrapper( N * imageWidth * imageWidth, images );
        CLIntWrapper *filterBuffer = cl->intWrapper( filterWidth * filterWidth, filter );
        CLIntWrapper *resultsBuffer = cl->intWrapper( N * imageWidth * imageWidth, results );
        imagesBuffer->copyToDevice();
        filterBuffer->copyToDevice();

        CLKernel *kernel = 0;
        kernel = cl->   buildKernel( "../ClConvolve.cl", "convolve_ints" );
        kernel->input( 1, &imageWidth );
        kernel->input( 1, &filterWidth );
        kernel->input( imagesBuffer );
        kernel->input( filterBuffer);
        kernel->output( resultsBuffer );
        int globalSize = N * imageWidth * imageWidth;
        int workgroupsize = cl->getMaxWorkgroupSize();
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
        kernel->run_1d( globalSize, workgroupsize );
        resultsBuffer->copyToHost();
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
           int *images, int *filters, int *results ) {
        OpenCLHelper *cl = new OpenCLHelper(0);

        CLIntWrapper *imagesBuffer = cl->intWrapper( numImages * numInputPlanes * imageWidth * imageWidth, images );
        CLIntWrapper *filterBuffer = cl->intWrapper( numFilters * numInputPlanes * filterWidth * filterWidth, filters );
        CLIntWrapper *resultsBuffer = cl->intWrapper( numImages * numFilters * imageWidth * imageWidth, results );
        imagesBuffer->copyToDevice();
        filterBuffer->copyToDevice();

        CLKernel *kernel = 0;
        kernel = cl->buildKernel( "../ClConvolve.cl", "convolve_imagecubes_int" );
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
        delete kernel;  
        delete cl;
        delete imagesBuffer;
        delete filterBuffer;
        delete resultsBuffer;
    }
};

