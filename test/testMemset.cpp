// tests the memset gpu kernel

#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "util/Timer.h"
#include "EasyCL.h"

using namespace std;

TEST( testMemset, basic ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    CLKernel *kMemset = 0;
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kMemset", "cl/memset.cl", "cl_memset", '""' )
    // ]]]
    // generated using cog, from cl/memset.cl:
    const char * kMemsetSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "kernel void cl_memset(global float *target, const float value, const int N) {\n"
    "    #define globalId get_global_id(0)\n"
    "    if ((int)globalId < N) {\n"
    "        target[globalId] = value;\n"
    "    }\n"
    "}\n"
    "\n"
    "";
    kMemset = cl->buildKernelFromString(kMemsetSource, "cl_memset", "", "cl/memset.cl");
    // [[[end]]]

    int N = 10000;
    float *myArray = new float[N];
    CLWrapper *myArrayWrapper = cl->wrap( N, myArray );
    myArrayWrapper->createOnDevice();
    kMemset->out( myArrayWrapper )->in( 99.0f )->in( N );
    int workgroupSize = 64;
    kMemset->run_1d( ( N + workgroupSize - 1 ) / workgroupSize * workgroupSize, workgroupSize );
    cl->finish();
    myArrayWrapper->copyToHost();
    for( int i = 0; i < 10; i++ ) {
//        cout << "myArray[" << i << "]=" << myArray[i] << endl;
    }
    for( int i = 0; i < N; i++ ) {
        EXPECT_EQ( 99.0f, myArray[i] );
    }

    delete kMemset;

    delete cl;
}

