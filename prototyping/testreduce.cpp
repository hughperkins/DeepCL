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
const int numBoards = 100000;
const int boardSize = 19;

// try using the binary reduce mechanism
// let's pretend we want to sum up 19x19 boards (which we kind of often do :-) )
// producing one value per board
TEST( testreduce, sumboards_cpu ) {
//    const int numBoards = 100000;
//    const int boardSize = 19;

    const int boardSizeSquared = boardSize * boardSize;

//    S timer;
    int totalLinearSize = numBoards * boardSize * boardSize;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
//    WeightRandomizer::randomize( input, totalLinearSize / 10 );
    for( int i = 0; i < totalLinearSize; i++ ) {
        input[i] = ( i % 231 ) / 231.0f;
    }
    float *sums = new float[numBoards];
    StatefulTimer::timeCheck("setup boards");

    for( int it = 0; it < numIts; it++ ) {
        for( int plane = 0; plane < numBoards; plane++ ) {
            float sum = 0;
            float *planeBoard = input + plane * boardSizeSquared;
            for( int i = 0; i < boardSizeSquared; i++ ) {
                sum += planeBoard[i];
            }
            sums[plane] = sum;
        }
        StatefulTimer::timeCheck("cpu time");
        // sum all, to ensure no compilation shortcutting
        float totalSum = 0;
        for( int i = 0; i < numBoards; i++ ) {
            totalSum += sums[i];
        }
        cout << "totalSum: " << totalSum << endl;
        StatefulTimer::timeCheck("totalsum");
    }
    StatefulTimer::dump(true);

    delete[] input;
    delete[] sums;

//    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    delete cl;
}

void setupBoards( OpenCLHelper *cl, CLWrapper *inputWrapper, int numBoards, int boardSize ) {
    CLKernel *kernel = cl->buildKernel( "../prototyping/testreduce.cl", "setupBoards" );
    kernel->inout( inputWrapper )->in( numBoards )->in( boardSize );
    int totalLinearSize = numBoards * boardSize * boardSize;
    int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    int numWorkgroups = ( totalLinearSize + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
    kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
    cl->finish();
    delete kernel;
}

float sumSums_singlethread( OpenCLHelper *cl, CLWrapper *sums, int numBoards ) {
    float sum[1];
    CLKernel *kernel = cl->buildKernel("../prototyping/testreduce.cl", "sumSums_singlethread" );
    kernel->in( sums )->out( 1, sum )->in( numBoards );
    kernel->run_1d( 1, 1 );
    cl->finish();
    return sum[0];
    delete kernel;
}

TEST( testreduce, sumboards_threadperboard ) {
    const int boardSizeSquared = boardSize * boardSize;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    Timer timer;
    int totalLinearSize = numBoards * boardSize * boardSize;
    cout << "totalLinearSize " << totalLinearSize << endl;
    cout << "memory " << ( totalLinearSize * 4 / 1024 / 1024 ) << "MB" << endl;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
    float *sums = new float[numBoards];

    CLWrapper *inputWrapper = cl->wrap( totalLinearSize, input );
    inputWrapper->createOnDevice();
    CLWrapper *sumsWrapper = cl->wrap( numBoards, sums );
    sumsWrapper->createOnDevice();

    setupBoards( cl, inputWrapper, numBoards, boardSize );

    StatefulTimer::timeCheck("setup boards");

    CLKernel *kernel = cl->buildKernel("../prototyping/testreduce.cl" ,"sum_threadperboard" );
    for( int it = 0; it < numIts; it++ ) {        
        kernel->in( inputWrapper )->out( sumsWrapper )->in( numBoards )->in( boardSize );
        int maxWorkgroupSize = cl->getMaxWorkgroupSize();
        int numWorkgroups = ( numBoards + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu time");

        // sum all, to ensure no compilation shortcutting
        float totalSum = sumSums_singlethread( cl, sumsWrapper, numBoards );
        cout << "totalSum: " << totalSum << endl;
        StatefulTimer::timeCheck("totalsum time" );
    }
    StatefulTimer::dump(true);

    delete kernel;

    delete inputWrapper;
    delete sumsWrapper;

    delete[] input;
    delete[] sums;

    delete cl;
}

TEST( testreduce, sumboards_threadperrow ) {
    const int boardSizeSquared = boardSize * boardSize;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

//    Timer timer;
    int totalLinearSize = numBoards * boardSize * boardSize;
    cout << "totalLinearSize " << totalLinearSize << endl;
    cout << "memory " << ( totalLinearSize * 4 / 1024 / 1024 ) << "MB" << endl;
    float *input = new float[ totalLinearSize ];
    StatefulTimer::timeCheck("allocated memory");
    float *sums = new float[numBoards];

    CLWrapper *inputWrapper = cl->wrap( totalLinearSize, input );
    inputWrapper->createOnDevice();
    float *rowsums = new float[numBoards * boardSize];
    CLWrapper *rowSumsWrapper = cl->wrap( numBoards * boardSize, rowsums );
    rowSumsWrapper->createOnDevice();
    CLWrapper *sumsWrapper = cl->wrap( numBoards, sums );
    sumsWrapper->createOnDevice();

    setupBoards( cl, inputWrapper, numBoards, boardSize );

    StatefulTimer::timeCheck("setup boards");

    CLKernel *kernel = cl->buildKernel("../prototyping/testreduce.cl" ,"sum_sumrow" );
    for( int it = 0; it < numIts; it++ ) {        
        kernel->in( inputWrapper )->out( rowSumsWrapper )->in( numBoards * boardSize )->in( boardSize );
        int maxWorkgroupSize = cl->getMaxWorkgroupSize();
        int numWorkgroups = ( numBoards * boardSize + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu time");

        kernel->in( rowSumsWrapper )->out( sumsWrapper )->in( numBoards )->in( boardSize );
        numWorkgroups = ( numBoards + maxWorkgroupSize - 1 ) / maxWorkgroupSize;
        kernel->run_1d( numWorkgroups * maxWorkgroupSize, maxWorkgroupSize );
        cl->finish();
        StatefulTimer::timeCheck("gpu 2 time");

        // sum all, to ensure no compilation shortcutting
        float totalSum = sumSums_singlethread( cl, sumsWrapper, numBoards );
        cout << "totalSum: " << totalSum << endl;
        StatefulTimer::timeCheck("totalsum time" );
    }
    StatefulTimer::dump(true);

    delete kernel;

    delete inputWrapper;
    delete sumsWrapper;

    delete[] input;
    delete[] sums;

    delete cl;
}

}

