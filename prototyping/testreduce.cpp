// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <iomanip>
#include <algorithm>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "test/WeightRandomizer.h"
#include "StatefulTimer.h"
#include "Timer.h"

#include "gtest/gtest.h"

using namespace std;

#include "test/gtest_supp.h"

namespace testreduce {

const int numIts = 10;
const int numImages = 100000;
const int imageSize = 19;

// try using the binary reduce mechanism
// let's pretend we want to sum up 19x19 images (which we kind of often do :-) )
// producing one value per image
TEST( testreduce, sumimages_cpu ) {
//    const int numImages = 100000;
//    const int imageSize = 19;

    const int imageSizeSquared = imageSize * imageSize;

//    S timer;
    int totalLinearSize = numImages * imageSize * imageSize;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
//    WeightRandomizer::randomize( input, totalLinearSize / 10 );
    for( int i = 0; i < totalLinearSize; i++ ) {
        input[i] = ( i % 231 ) / 231.0f;
    }
    float *sums = new float[numImages];
    StatefulTimer::timeCheck("setup images");

    float totalSum = 0;
    for( int it = 0; it < numIts; it++ ) {
        for( int plane = 0; plane < numImages; plane++ ) {
            float sum = 0;
            float *planeImage = input + plane * imageSizeSquared;
            for( int i = 0; i < imageSizeSquared; i++ ) {
                sum += planeImage[i];
            }
            sums[plane] = sum;
        }
        StatefulTimer::timeCheck("cpu time");
        // sum all, to ensure no compilation shortcutting
        for( int i = 0; i < numImages; i++ ) {
            totalSum += sums[i];
        }
        StatefulTimer::timeCheck("totalsum");
    }
    cout << "totalSum: " << totalSum << endl;
    StatefulTimer::dump(true);

    delete[] input;
    delete[] sums;

//    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    delete cl;
}

void setupImages( OpenCLHelper *cl, CLWrapper *inputWrapper, int numImages, int imageSize ) {
    CLKernel *kernel = cl->buildKernel( "../prototyping/testreduce.cl", "setupImages" );
    kernel->inout( inputWrapper )->in( numImages )->in( imageSize );
    int totalLinearSize = numImages * imageSize * imageSize;
    int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    int numWorkgroups = ( totalLinearSize + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
    kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
    cl->finish();
    delete kernel;
}

float sumSums_singlethread( OpenCLHelper *cl, CLWrapper *sums, int numImages ) {
    float sum[1];
    CLKernel *kernel = cl->buildKernel("../prototyping/testreduce.cl", "sumSums_singlethread" );
    kernel->in( sums )->out( 1, sum )->in( numImages );
    kernel->run_1d( 1, 1 );
    cl->finish();
    return sum[0];
    delete kernel;
}

TEST( testreduce, sumimages_threadperimage ) {
//    const int imageSizeSquared = imageSize * imageSize;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    Timer timer;
    int totalLinearSize = numImages * imageSize * imageSize;
//    cout << "totalLinearSize " << totalLinearSize << endl;
//    cout << "memory " << ( totalLinearSize * 4 / 1024 / 1024 ) << "MB" << endl;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
    float *sums = new float[numImages];

    CLWrapper *inputWrapper = cl->wrap( totalLinearSize, input );
    inputWrapper->createOnDevice();
    CLWrapper *sumsWrapper = cl->wrap( numImages, sums );
    sumsWrapper->createOnDevice();

    setupImages( cl, inputWrapper, numImages, imageSize );

    StatefulTimer::timeCheck("setup images");

    CLKernel *kernel = cl->buildKernel("../prototyping/testreduce.cl" ,"sum_threadperimage" );
    float totalSum = 0;
    for( int it = 0; it < numIts; it++ ) {        
        kernel->in( inputWrapper )->out( sumsWrapper )->in( numImages )->in( imageSize );
        int maxWorkgroupSize = cl->getMaxWorkgroupSize();
        int numWorkgroups = ( numImages + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu time");

        // sum all, to ensure no compilation shortcutting
        totalSum += sumSums_singlethread( cl, sumsWrapper, numImages );
        StatefulTimer::timeCheck("totalsum time" );
    }
    cout << "totalSum: " << totalSum << endl;
    StatefulTimer::dump(true);

    delete kernel;

    delete inputWrapper;
    delete sumsWrapper;

    delete[] input;
    delete[] sums;

    delete cl;
}

TEST( testreduce, sumimages_threadperrow ) {
//    const int imageSizeSquared = imageSize * imageSize;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    Timer timer;
    int totalLinearSize = numImages * imageSize * imageSize;
//    cout << "totalLinearSize " << totalLinearSize << endl;
//    cout << "memory " << ( totalLinearSize * 4 / 1024 / 1024 ) << "MB" << endl;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
    float *sums = new float[numImages];

    CLWrapper *inputWrapper = cl->wrap( totalLinearSize, input );
    inputWrapper->createOnDevice();
    float *rowsums = new float[numImages * imageSize];
    CLWrapper *rowSumsWrapper = cl->wrap( numImages * imageSize, rowsums );
    rowSumsWrapper->createOnDevice();
    CLWrapper *sumsWrapper = cl->wrap( numImages, sums );
    sumsWrapper->createOnDevice();

    setupImages( cl, inputWrapper, numImages, imageSize );

    StatefulTimer::timeCheck("setup images");

    CLKernel *kernel = cl->buildKernel("../prototyping/testreduce.cl" ,"sum_sumrow" );
    float totalSum = 0;
    for( int it = 0; it < numIts; it++ ) {        
        kernel->in( inputWrapper )->out( rowSumsWrapper )->in( numImages * imageSize )->in( imageSize );
        int maxWorkgroupSize = cl->getMaxWorkgroupSize();
        int numWorkgroups = ( numImages * imageSize + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu time");

        kernel->in( rowSumsWrapper )->out( sumsWrapper )->in( numImages )->in( imageSize );
        numWorkgroups = ( numImages + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu 2 time");

        // sum all, to ensure no compilation shortcutting
        totalSum += sumSums_singlethread( cl, sumsWrapper, numImages );
        StatefulTimer::timeCheck("totalsum time" );
    }
    cout << "totalSum: " << totalSum << endl;
    StatefulTimer::dump(true);

    delete kernel;

    delete inputWrapper;
    delete sumsWrapper;

    delete[] input;
    delete[] sums;

    delete cl;
}

TEST( testreduce, sumimages_workgroupperimage_threadperrow ) {
//    const int imageSizeSquared = imageSize * imageSize;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    Timer timer;
    int totalLinearSize = numImages * imageSize * imageSize;
//    cout << "totalLinearSize " << totalLinearSize << endl;
//    cout << "memory " << ( totalLinearSize * 4 / 1024 / 1024 ) << "MB" << endl;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
    float *sums = new float[numImages];

    CLWrapper *inputWrapper = cl->wrap( totalLinearSize, input );
    inputWrapper->createOnDevice();
    float *rowsums = new float[numImages * imageSize];
    CLWrapper *rowSumsWrapper = cl->wrap( numImages * imageSize, rowsums );
    rowSumsWrapper->createOnDevice();
    CLWrapper *sumsWrapper = cl->wrap( numImages, sums );
    sumsWrapper->createOnDevice();

    setupImages( cl, inputWrapper, numImages, imageSize );

    StatefulTimer::timeCheck("setup images");

    CLKernel *kernel1 = cl->buildKernel("../prototyping/testreduce.cl" ,"sum_workgroupperimage_sumrow" );
    CLKernel *kernel2 = cl->buildKernel("../prototyping/testreduce.cl" ,"sum_sumrow" );
    float totalSum = 0;
    for( int it = 0; it < numIts; it++ ) {        
        kernel1->in( inputWrapper )->out( rowSumsWrapper )->in( imageSize );
        int workgroupSize = imageSize;
        int numWorkgroups = numImages;
        kernel1->run_1d( numWorkgroups * workgroupSize, workgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu time");

        int maxWorkgroupSize = cl->getMaxWorkgroupSize();
        kernel2->in( rowSumsWrapper )->out( sumsWrapper )->in( numImages )->in( imageSize );
        numWorkgroups = ( numImages + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        kernel2->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu 2 time");

        // sum all, to ensure no compilation shortcutting
        totalSum += sumSums_singlethread( cl, sumsWrapper, numImages );
        StatefulTimer::timeCheck("totalsum time" );
    }
    cout << "totalSum: " << totalSum << endl;
    StatefulTimer::dump(true);

    delete kernel1;
    delete kernel2;

    delete inputWrapper;
    delete sumsWrapper;

    delete[] input;
    delete[] sums;

    delete cl;
}

TEST( testreduce, sum_workgroupperimage_threadperpixel ) {
//    const int imageSizeSquared = imageSize * imageSize;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    Timer timer;
    int totalLinearSize = numImages * imageSize * imageSize;
//    cout << "totalLinearSize " << totalLinearSize << endl;
//    cout << "memory " << ( totalLinearSize * 4 / 1024 / 1024 ) << "MB" << endl;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
    float *sums = new float[numImages];

    CLWrapper *inputWrapper = cl->wrap( totalLinearSize, input );
    inputWrapper->createOnDevice();
//    float *rowsums = new float[numImages * imageSize];
//    CLWrapper *rowSumsWrapper = cl->wrap( numImages * imageSize, rowsums );
//    rowSumsWrapper->createOnDevice();
    CLWrapper *sumsWrapper = cl->wrap( numImages, sums );
    sumsWrapper->createOnDevice();

    setupImages( cl, inputWrapper, numImages, imageSize );

    StatefulTimer::timeCheck("setup images");

    CLKernel *kernel = cl->buildKernel("../prototyping/testreduce.cl" ,"sum_workgroupperimage_threadperpixel" );
    float totalSum = 0;
    for( int it = 0; it < numIts; it++ ) {     
        setupImages( cl, inputWrapper, numImages, imageSize );
        StatefulTimer::timeCheck("setup images");

        // ( global float *images, global float *sums, const int numImages, const int imageSize   
        kernel->in( inputWrapper )->out( sumsWrapper )->in( numImages )->in( imageSize )
            ->in( OpenCLHelper::getNextPower2( imageSize * imageSize ) );
        kernel->run_1d( numImages * imageSize * imageSize, imageSize * imageSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu time");

//        kernel->in( rowSumsWrapper )->out( sumsWrapper )->in( numImages )->in( imageSize );
//        numWorkgroups = ( numImages + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
//        kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
//        cl->finish();
//        StatefulTimer::timeCheck("gpu 2 time");

        // sum all, to ensure no compilation shortcutting
        totalSum += sumSums_singlethread( cl, sumsWrapper, numImages );
        StatefulTimer::timeCheck("totalsum time" );
    }
    cout << "totalSum: " << totalSum << endl;
    StatefulTimer::dump(true);

    delete kernel;

    delete inputWrapper;
    delete sumsWrapper;

    delete[] input;
    delete[] sums;

    delete cl;
}

TEST( testreduce, sum_workgroupperimage_threadperpixel_local ) {
//    const int imageSizeSquared = imageSize * imageSize;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    Timer timer;
    int totalLinearSize = numImages * imageSize * imageSize;
//    cout << "totalLinearSize " << totalLinearSize << endl;
//    cout << "memory " << ( totalLinearSize * 4 / 1024 / 1024 ) << "MB" << endl;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
    float *sums = new float[numImages];

    CLWrapper *inputWrapper = cl->wrap( totalLinearSize, input );
    inputWrapper->createOnDevice();
//    float *rowsums = new float[numImages * imageSize];
//    CLWrapper *rowSumsWrapper = cl->wrap( numImages * imageSize, rowsums );
//    rowSumsWrapper->createOnDevice();
    CLWrapper *sumsWrapper = cl->wrap( numImages, sums );
    sumsWrapper->createOnDevice();

    setupImages( cl, inputWrapper, numImages, imageSize );

    StatefulTimer::timeCheck("setup images");

    CLKernel *kernel = cl->buildKernel("../prototyping/testreduce.cl" ,"sum_workgroupperimage_threadperpixel_local" );
    float totalSum = 0;
    for( int it = 0; it < numIts; it++ ) {     
        setupImages( cl, inputWrapper, numImages, imageSize );
        StatefulTimer::timeCheck("setup images");

        // ( global float *images, global float *sums, const int numImages, const int imageSize   
        kernel->in( inputWrapper )->out( sumsWrapper )->localFloats(imageSize * imageSize)->in( numImages )->in( imageSize )
            ->in( OpenCLHelper::getNextPower2( imageSize * imageSize ) );
        kernel->run_1d( numImages * imageSize * imageSize, imageSize * imageSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu time");

//        kernel->in( rowSumsWrapper )->out( sumsWrapper )->in( numImages )->in( imageSize );
//        numWorkgroups = ( numImages + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
//        kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
//        cl->finish();
//        StatefulTimer::timeCheck("gpu 2 time");

        // sum all, to ensure no compilation shortcutting
        totalSum += sumSums_singlethread( cl, sumsWrapper, numImages );
        StatefulTimer::timeCheck("totalsum time" );
    }
    cout << "totalSum: " << totalSum << endl;
//    cout << endl;
    StatefulTimer::dump(true);

    delete kernel;

    delete inputWrapper;
    delete sumsWrapper;

    delete[] input;
    delete[] sums;

    delete cl;
}

}

