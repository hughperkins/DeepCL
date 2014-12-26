#pragma once

#include "OpenCLHelper.h"

template< typename T>
class ClConvolve2 {
    int numImages;
    int numPlanes;
    int imageLengthSide;
    T *images;
    CLWrapper *imagesBuffer;
//    T *r;
    OpenCLHelper *cl;
    CLKernel *kernel;
public:
    ClConvolve2( int numImages, int numPlanes, int imageLengthSide, T *inputCubes ) {
         this->numImages = numImages;
         this->numPlanes = numPlanes;
         this->inputCubes = inputCubes;
         this->imageLengthSide = imageLengthSide;
         this->cl = new OpenCLHelper(0);
         this->kernel = cl->buildKernel( "ClConvolve.cl", getKernelName() );
         this->images = inputCubes;
         this->imagesBuffer = cl->wrap( numImages * numPlanes * imageLengthSide * imageLengthSide, inputCubes );
    }
    ~ClConvolve2() {
         if( imagesBuffer != 0 ) {
              delete imagesBuffer;
         }
         delete kernel;
         delete cl;
    }
    void convolve( int numFilters, int filterLengthSide, T *filterStack ) {
        T *results = new T[numImages * numFilters * imageLengthSide * imageLengthSide];
        CLWrapper *resultsBuffer = cl->wrap( numImages * numFilters * imageLengthSide * imageLengthSide, results );
        CLWrapper *filterBuffer = cl->wrap( numFilters * numPlanes * filterLengthSide * filterLengthSide, filterStack );
        imagesBuffer->copyToDevice();
        filterBuffer->copyToDevice();
//        timer.timeCheck("copied data to device");

        kernel->input( 1, &numPlanes );
        kernel->input( 1, &numFilters );
        kernel->input( 1, &imageLengthSide );
        kernel->input( 1, &filterLengthSide );
        kernel->input( imagesBuffer );
        kernel->input( filterBuffer);
        kernel->output( resultsBuffer );
        int globalSize = numImages * numFilters * imageLengthSide * imageLengthSide;
        int workgroupsize = cl->getMaxWorkgroupSize();
        globalSize = ( ( globalSize + workgroupsize - 1 ) / workgroupsize ) * workgroupsize;
        kernel->run_1d( globalSize, workgroupsize );
        //resultsBuffer->copyToHost();
        delete filterBuffer;
    }
    T *getResults() {
        resultsBuffer->copyToHost();
        return results;
    }
    std::string getKernelName();
};

template<>
std::string ClConvolve2<int>::getKernelName(){ return "convolve_imagecubes_int"; }

template<>
std::string ClConvolve2<float>::getKernelName(){ return "convolve_imagecubes_float"; }

