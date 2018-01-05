// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "pooling/PoolingBackward.h"
#include "pooling/PoolingForward.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/TestArgsParser.h"
#include "test/WeightRandomizer.h"

using namespace std;

TEST( testpoolingbackward, basic ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 4;
    int poolingSize = 2;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    PoolingBackward *poolingBackprop = PoolingBackward::instanceForTest( cl, false, numPlanes, imageSize, poolingSize );
    float errors[] = {
        3, 5,
        2, 9
    };
    int selectors[] = {
        2, 1,
        0, 3
    };
    float *errorsForUpstream = new float[ poolingBackprop->getInputNumElements( batchSize ) ];

    poolingBackprop->backward( batchSize, errors, selectors, errorsForUpstream );

//    float *expectedErrorsForUpstream = new float[ poolingForward->getInputNumElements( batchSize ) ];
//    memset( expectedErrorsForUpstream, 0, sizeof(float) * poolingForward->getInputNumElements( batchSize ) ];
    float expectedErrorsForUpstream[] = {
        0,0,0,5,
        3,0,0,0,
        2,0,0,0,
        0,0,0,9,
    };
    for( int i = 0; i < 16; i++ ) {
        ASSERT_EQ( expectedErrorsForUpstream[i], errorsForUpstream[i] );
    }

    delete poolingBackprop;
    delete[] errorsForUpstream;
    delete cl;
}

TEST( testpoolingbackward, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    int poolingSize = 2;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    PoolingBackward *poolingBackprop = PoolingBackward::instanceForTest( cl, false, numPlanes, imageSize, poolingSize );
    float errors[] = {
        3, 
        5,
        2, 
        9
    };
    int selectors[] = {
        2, 
        1,
        0, 
        3
    };
    float *errorsForUpstream = new float[ poolingBackprop->getInputNumElements( batchSize ) ];

    poolingBackprop->backward( batchSize, errors, selectors, errorsForUpstream );

//    float *expectedErrorsForUpstream = new float[ poolingForward->getInputNumElements( batchSize ) ];
//    memset( expectedErrorsForUpstream, 0, sizeof(float) * poolingForward->getInputNumElements( batchSize ) ];
    float expectedErrorsForUpstream[] = {
        0,0,
        3,0,

        0,5,
        0,0,

        2,0,
        0,0,
        
        0,0,
        0,9,
    };
    for( int i = 0; i < 16; i++ ) {
        ASSERT_EQ( expectedErrorsForUpstream[i], errorsForUpstream[i] );
    }

    delete poolingBackprop;
    delete[] errorsForUpstream;
    delete cl;
}

TEST( SLOW_testpoolingbackward, compare_args ) {
    int inputSize = 9;
    int poolingSize = 2;
    int instance0 = 0;
    int instance1 = 1;
    int numPlanes = 4;
    int batchSize = 6;
    int its = 3;
    TestArgsParser::arg( "its", &its );
    TestArgsParser::arg( "batchSize", &batchSize );
    TestArgsParser::arg( "poolingsize", &poolingSize );
    TestArgsParser::arg( "numplanes", &numPlanes );
    TestArgsParser::arg( "inputimagesize", &inputSize );
    TestArgsParser::arg( "instance0", &instance0 );
    TestArgsParser::arg( "instance1", &instance1 );
    TestArgsParser::go();

    bool padZeros = true;

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    PoolingBackward *p0 = PoolingBackward::instanceSpecific( instance0, cl, padZeros, numPlanes, inputSize, poolingSize );
    PoolingBackward *p1 = PoolingBackward::instanceSpecific( instance1, cl, padZeros, numPlanes, inputSize, poolingSize );
    int outputSize = p1->outputSize;
    int errorsSize = batchSize * outputSize * outputSize * numPlanes;
    float *errors = new float[ errorsSize ];
    int inputNumElements = batchSize * inputSize * inputSize * numPlanes;
    int *selectors = new int[ errorsSize ];
    float *errorsForUpstream0 = new float[ inputNumElements ];
    float *errorsForUpstream1 = new float[ inputNumElements ];
    
    PoolingForward *forwardprop = PoolingForward::instanceSpecific( 0, cl, padZeros, numPlanes, inputSize, poolingSize );
    float *output = new float[errorsSize];
    float *input = new float[inputNumElements];
    float *errorsForUpstream[2];
    errorsForUpstream[0] = errorsForUpstream0;
    errorsForUpstream[1] = errorsForUpstream1;
    PoolingBackward *props[2];
    props[0] = p0;
    props[1] = p1;
    for( int it = 0; it < its; it++ ) {
        // selectors might go over the edge if we just choose random ints
        // easiest way to select valid selectors might be to just forwardforward first?

        WeightRandomizer::randomize( it, errors, errorsSize, -0.1f, 0.1f );
        WeightRandomizer::randomize( it, input, inputNumElements, -0.1f, 0.1f );    
        forwardprop->forward( batchSize, input, selectors, output );

        for( int instance = 0; instance < 2; instance++ ) {
            props[instance]->backward( batchSize, errors, selectors, errorsForUpstream[instance] );
        }
        bool ok = true;
        int numErrors = 0;
        for( int i = 0; i < inputNumElements; i++ ) {
            if( errorsForUpstream0[i] != errorsForUpstream1[i] ) {
                cout << "diff: i=" << i << " " << errorsForUpstream0[i] << " != " << errorsForUpstream1[i] << endl;
                ok = false;
                numErrors++;
                if( numErrors > 20 ) {
                    cout << " ... etc ...." << endl;
                    break;
                }
            }
        }
        EXPECT_EQ( true, ok );
        if( !ok ) {
            cout << " breaking after " << it << " its, because of FAIL errors" << endl;
            break; // no point in continuing...
        }
    }

    delete forwardprop;
    delete[] input;
    delete[] output;
    delete[] errors;
    delete[] selectors;
    delete[] errorsForUpstream0;
    delete[] errorsForUpstream1;
    delete p0;
    delete p1;
    delete cl;
}

/*
TEST( testpoolingforward, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    int poolingSize = 2;
    EasyCL cl;
    PoolingForward *poolingForward = PoolingForward::instanceForTest( cl, numPlanes, imageSize, poolingSize );
    float data[] = { 1, 2, 
                    5, 3,

                     3, 8, 
                    4, 1,

                     3, 33, 
                    14,23,

                     -1, -3.5f,
                    37.4f,5
    };
    int outputNumElements = poolingForward->getOutputNumElements( batchSize );
    int *selectors = new int[outputNumElements];
    float *output = new float[outputNumElements];

    poolingForward->forward( batchSize, data, selectors, output );

    EXPECT_EQ( selectors[0], 2 );
    EXPECT_EQ( selectors[1], 1 );
    EXPECT_EQ( selectors[2], 1 );
    EXPECT_EQ( selectors[3], 2 );

    EXPECT_EQ( output[0], 5 );
    EXPECT_EQ( output[1], 8 );
    EXPECT_EQ( output[2], 33 );
    EXPECT_EQ( output[3], 37.4f );

    delete poolingForward;
    delete[] selectors;
    delete[] output;
}
*/

