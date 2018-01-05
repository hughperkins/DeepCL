// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "dropout/DropoutBackward.h"
#include "dropout/DropoutForward.h"
//#include "DropoutFunction.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/TestArgsParser.h"
#include "test/WeightRandomizer.h"

using namespace std;

TEST( testdropoutbackward, basic ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 3;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    DropoutBackward *dropoutBackprop = DropoutBackward::instanceForTest( cl, numPlanes, imageSize, 0.6f );
    uchar mask[] = {
        1,1,0,
        0,1,1,
        1,0,1
    };
    float errors[] = {
        3, 5,-2.7f,
        2, -9, 2.1f,
        0, -1.1f, 3.5f
    };
    int inputTotalSize = dropoutBackprop->getInputNumElements( batchSize );
    EXPECT_EQ( batchSize * imageSize * imageSize, inputTotalSize );
    float *errorsForUpstream = new float[ inputTotalSize ];

    dropoutBackprop->backward( batchSize, mask, errors, errorsForUpstream );

    EXPECT_FLOAT_NEAR( 3, errorsForUpstream[0] );
    EXPECT_FLOAT_NEAR( 5, errorsForUpstream[1] );
    EXPECT_FLOAT_NEAR( 0, errorsForUpstream[2] );

    EXPECT_FLOAT_NEAR( 0, errorsForUpstream[3] );
    EXPECT_FLOAT_NEAR( -9, errorsForUpstream[4] );
    EXPECT_FLOAT_NEAR( 2.1f, errorsForUpstream[5] );

    EXPECT_FLOAT_NEAR( 0, errorsForUpstream[6] );
    EXPECT_FLOAT_NEAR( 0, errorsForUpstream[7] );
    EXPECT_FLOAT_NEAR( 3.5f, errorsForUpstream[8] );
//    for( int i = 0; i < 16; i++ ) {
//        EXPECT_FLOAT_NEAR( expectedErrorsForUpstream[i], errorsForUpstream[i] );
//    }

    delete dropoutBackprop;
    delete[] errorsForUpstream;
    delete cl;
}

TEST( testdropoutbackward, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    DropoutBackward *dropoutBackprop = DropoutBackward::instanceForTest( cl, numPlanes, imageSize, 0.6f );
    uchar mask[] = {
        1,
        1,
        0,
        1
    };
    float errors[] = {
        3, 
        5,
        2, 
        9
    };
    float *errorsForUpstream = new float[ dropoutBackprop->getInputNumElements( batchSize ) ];

    dropoutBackprop->backward( batchSize, mask, errors, errorsForUpstream );

//    float *expectedErrorsForUpstream = new float[ dropoutForward->getInputNumElements( batchSize ) ];
//    memset( expectedErrorsForUpstream, 0, sizeof(float) * dropoutForward->getInputNumElements( batchSize ) ];
    float expectedErrorsForUpstream[] = {
        3,
        5,
        0,
        9
    };
    for( int i = 0; i < 4; i++ ) {
        ASSERT_EQ( expectedErrorsForUpstream[i], errorsForUpstream[i] );
    }

    delete dropoutBackprop;
    delete[] errorsForUpstream;
    delete cl;
}

TEST( testdropoutbackward, compare_args ) {
    int inputSize = 9;
    float dropRatio = 0.6f;
    int instance0 = 0;
    int instance1 = 1;
    int numPlanes = 4;
    int batchSize = 6;
    int its = 3;
    TestArgsParser::arg( "its", &its );
    TestArgsParser::arg( "batchSize", &batchSize );
    TestArgsParser::arg( "dropratio", &dropRatio );
//    TestArgsParser::arg( "dropoutsize", &dropoutSize );
    TestArgsParser::arg( "numplanes", &numPlanes );
    TestArgsParser::arg( "inputimagesize", &inputSize );
    TestArgsParser::arg( "instance0", &instance0 );
    TestArgsParser::arg( "instance1", &instance1 );
    TestArgsParser::go();

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    DropoutBackward *p0 = DropoutBackward::instanceSpecific( instance0, cl, numPlanes, inputSize, dropRatio );
    DropoutBackward *p1 = DropoutBackward::instanceSpecific( instance1, cl, numPlanes, inputSize, dropRatio );
    int outputSize = p1->outputSize;
    int errorsSize = batchSize * outputSize * outputSize * numPlanes;
    float *errors = new float[ errorsSize ];
    int inputNumElements = batchSize * inputSize * inputSize * numPlanes;
    float *errorsForUpstream0 = new float[ inputNumElements ];
    float *errorsForUpstream1 = new float[ inputNumElements ];
    
    DropoutForward *forwardprop = DropoutForward::instanceSpecific( 0, cl, numPlanes, inputSize, dropRatio );
    float *input = new float[inputNumElements];
    float *output = new float[errorsSize];
    uchar *mask = new uchar[inputNumElements];
    float *errorsForUpstream[2];
    errorsForUpstream[0] = errorsForUpstream0;
    errorsForUpstream[1] = errorsForUpstream1;
    DropoutBackward *props[2];
    props[0] = p0;
    props[1] = p1;
    for( int it = 0; it < its; it++ ) {
        // selectors might go over the edge if we just choose random ints
        // easiest way to select valid selectors might be to just forwardforward first?

        WeightRandomizer::randomize( it + 1, errors, errorsSize, -0.1f, 0.1f );
        WeightRandomizer::randomize( it + 1, input, inputNumElements, -0.1f, 0.1f );
        WeightRandomizer::randomizeInts( it + 1, mask, inputNumElements, 0, 2 );
        forwardprop->forward( batchSize, mask, input, output );

        for( int instance = 0; instance < 2; instance++ ) {
            props[instance]->backward( batchSize, mask, errors, errorsForUpstream[instance] );
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
        EXPECT_FLOAT_NEAR( true, ok );
        if( !ok ) {
            cout << " breaking after " << it << " its, because of FAIL errors" << endl;
            break; // no point in continuing...
        }
    }

    delete forwardprop;
    delete[] input;
    delete[] output;
    delete[] errors;
    delete[] errorsForUpstream0;
    delete[] errorsForUpstream1;
    delete p0;
    delete p1;
    delete cl;
}

/*
TEST( testdropoutforward, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    int dropoutSize = 2;
    EasyCL cl;
    DropoutForward *dropoutForward = DropoutForward::instanceForTest( cl, numPlanes, imageSize, dropoutSize );
    float data[] = { 1, 2, 
                    5, 3,

                     3, 8, 
                    4, 1,

                     3, 33, 
                    14,23,

                     -1, -3.5f,
                    37.4f,5
    };
    int outputNumElements = dropoutForward->getOutputNumElements( batchSize );
    int *selectors = new int[outputNumElements];
    float *output = new float[outputNumElements];

    dropoutForward->forward( batchSize, data, selectors, output );

    EXPECT_FLOAT_NEAR( selectors[0], 2 );
    EXPECT_FLOAT_NEAR( selectors[1], 1 );
    EXPECT_FLOAT_NEAR( selectors[2], 1 );
    EXPECT_FLOAT_NEAR( selectors[3], 2 );

    EXPECT_FLOAT_NEAR( output[0], 5 );
    EXPECT_FLOAT_NEAR( output[1], 8 );
    EXPECT_FLOAT_NEAR( output[2], 33 );
    EXPECT_FLOAT_NEAR( output[3], 37.4f );

    delete dropoutForward;
    delete[] selectors;
    delete[] output;
}
*/

