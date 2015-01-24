// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OpenCLHelper.h"

#include "PoolingPropagate.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

using namespace std;

TEST( testpoolingpropagate, basic ) {
    int batchSize = 1;
    int numPlanes = 1;
    int boardSize = 4;
    int poolingSize = 2;
    OpenCLHelper cl;
    PoolingPropagate *poolingPropagate = PoolingPropagate::instanceForTest( &cl, numPlanes, boardSize, poolingSize );
    float data[] = { 1, 2, 5, 3,
                     3, 8, 4, 1,
                     3, 33, 14,23,
                     -1, -3.5f,37.4f,5
    };
    float *output = poolingPropagate->propagate( batchSize, data );
    EXPECT_EQ( output[0], 8 );
    EXPECT_EQ( output[1], 5 );
    EXPECT_EQ( output[2], 33 );
    EXPECT_EQ( output[3], 37.4f );
    delete poolingPropagate;
}

TEST( testpoolingpropagate, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int boardSize = 2;
    int poolingSize = 2;
    OpenCLHelper cl;
    PoolingPropagate *poolingPropagate = PoolingPropagate::instanceForTest( &cl, numPlanes, boardSize, poolingSize );
    float data[] = { 1, 2, 
                    5, 3,

                     3, 8, 
                    4, 1,

                     3, 33, 
                    14,23,

                     -1, -3.5f,
                    37.4f,5
    };
    float *output = poolingPropagate->propagate( batchSize, data );
    EXPECT_EQ( output[0], 5 );
    EXPECT_EQ( output[1], 8 );
    EXPECT_EQ( output[2], 33 );
    EXPECT_EQ( output[3], 37.4f );
    delete poolingPropagate;
}

