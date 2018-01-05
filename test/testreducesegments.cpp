 // Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include <iostream>
#include <string>
#include <algorithm>

#include "EasyCL.h"

#include "conv/ReduceSegments.h"

using namespace std;

TEST( testreducesegments, basic ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    ReduceSegments *reduceSegments = new ReduceSegments( cl );

    int segmentLength = 17;
    int numSegments = 210;
    int N = segmentLength * numSegments;
    float *data = new float[ N * 2 ];

    for( int i = 0; i < N * 2; i++ ) {
        data[i] = 15 + 2 * i;
    }
    CLWrapper *inWrapper = cl->wrap( N, data );
    inWrapper->copyToDevice();
    float *out = new float[ numSegments ];
    CLWrapper *outWrapper = cl->wrap( numSegments, out );
    outWrapper->createOnDevice();
    reduceSegments->reduce( N, segmentLength, inWrapper, outWrapper );
    outWrapper->copyToHost();
    for( int i = 0; i < numSegments; i++ ) {
        float sum = 0;
        for( int j = 0; j < segmentLength; j++ ) {
            sum += data[ i * segmentLength + j ];
        }
        EXPECT_EQ( sum, out[i] );
    }

    delete outWrapper;
    delete inWrapper;
    delete[] out;
    delete[] data;
    delete reduceSegments;
    delete cl;
}

