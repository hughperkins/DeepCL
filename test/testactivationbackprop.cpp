// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "OpenCLHelper.h"

#include "ActivationBackprop.h"
#include "ActivationPropagate.h"
#include "ActivationFunction.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/TestArgsParser.h"
#include "test/WeightRandomizer.h"

using namespace std;

TEST( testactivationbackprop, basic ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 3;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    ActivationBackprop *activationBackprop = ActivationBackprop::instanceForTest( cl, numPlanes, imageSize, new ReluActivation() );
    float inputs[] = {
        1, -0.1f, 0.1f,
        0.5f, -1000, 1000,
        2.5f, 2.0f, -1.0f
    };
    float errors[] = {
        3, 5,-2.7f,
        2, -9, 2.1f,
        0, -1.1f, 3.5f
    };
    int inputTotalSize = activationBackprop->getInputSize( batchSize );
    EXPECT_EQ( batchSize * imageSize * imageSize, inputTotalSize );
    float *errorsForUpstream = new float[ inputTotalSize ];

    activationBackprop->backpropErrors( batchSize, inputs, errors, errorsForUpstream );

//    float *expectedErrorsForUpstream = new float[ activationPropagate->getInputSize( batchSize ) ];
//    memset( expectedErrorsForUpstream, 0, sizeof(float) * activationPropagate->getInputSize( batchSize ) ];
    float expectedErrorsForUpstream[] = {
        3,0,-2.7f,
        2,0,2.1f,
        0,-1.1f,0,
    };
    EXPECT_EQ( 3, errorsForUpstream[0] );
    EXPECT_EQ( 0, errorsForUpstream[1] );
    EXPECT_EQ( -2.7f, errorsForUpstream[2] );

    EXPECT_EQ( 2, errorsForUpstream[3] );
    EXPECT_EQ( 0, errorsForUpstream[4] );
    EXPECT_EQ( 2.1f, errorsForUpstream[5] );

    EXPECT_EQ( 0, errorsForUpstream[6] );
    EXPECT_EQ( -1.1f, errorsForUpstream[7] );
    EXPECT_EQ( 0, errorsForUpstream[8] );
//    for( int i = 0; i < 16; i++ ) {
//        EXPECT_EQ( expectedErrorsForUpstream[i], errorsForUpstream[i] );
//    }

    delete activationBackprop;
    delete[] errorsForUpstream;
    delete cl;
}

TEST( testactivationbackprop, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    ActivationBackprop *activationBackprop = ActivationBackprop::instanceForTest( cl, numPlanes, imageSize, new ReluActivation() );
    float inputs[] = {
        2,
        -1,
        -2,
        2
    };
    float errors[] = {
        3, 
        5,
        2, 
        9
    };
    float *errorsForUpstream = new float[ activationBackprop->getInputSize( batchSize ) ];

    activationBackprop->backpropErrors( batchSize, inputs, errors, errorsForUpstream );

//    float *expectedErrorsForUpstream = new float[ activationPropagate->getInputSize( batchSize ) ];
//    memset( expectedErrorsForUpstream, 0, sizeof(float) * activationPropagate->getInputSize( batchSize ) ];
    float expectedErrorsForUpstream[] = {
        3,
        0,
        0,
        9
    };
    for( int i = 0; i < 4; i++ ) {
        ASSERT_EQ( expectedErrorsForUpstream[i], errorsForUpstream[i] );
    }

    delete activationBackprop;
    delete[] errorsForUpstream;
    delete cl;
}

TEST( SLOW_testactivationbackprop, compare_args ) {
    int inputImageSize = 9;
    std::string activation = "relu";
    int instance0 = 0;
    int instance1 = 1;
    int numPlanes = 4;
    int batchSize = 6;
    int its = 3;
    TestArgsParser::arg( "its", &its );
    TestArgsParser::arg( "batchSize", &batchSize );
    TestArgsParser::arg( "activation", &activation );
//    TestArgsParser::arg( "activationsize", &activationSize );
    TestArgsParser::arg( "numplanes", &numPlanes );
    TestArgsParser::arg( "inputimagesize", &inputImageSize );
    TestArgsParser::arg( "instance0", &instance0 );
    TestArgsParser::arg( "instance1", &instance1 );
    TestArgsParser::go();

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    ActivationBackprop *p0 = ActivationBackprop::instanceSpecific( instance0, cl, numPlanes, inputImageSize, ActivationFunction::fromName( activation ) );
    ActivationBackprop *p1 = ActivationBackprop::instanceSpecific( instance1, cl, numPlanes, inputImageSize, ActivationFunction::fromName( activation ) );
    int outputImageSize = p1->outputImageSize;
    int errorsSize = batchSize * outputImageSize * outputImageSize * numPlanes;
    float *errors = new float[ errorsSize ];
    int inputSize = batchSize * inputImageSize * inputImageSize * numPlanes;
    float *errorsForUpstream0 = new float[ inputSize ];
    float *errorsForUpstream1 = new float[ inputSize ];
    
    ActivationPropagate *forwardprop = ActivationPropagate::instanceSpecific( 0, cl, numPlanes, inputImageSize, ActivationFunction::fromName( activation ) );
    float *output = new float[errorsSize];
    float *input = new float[inputSize];
    float *errorsForUpstream[2];
    errorsForUpstream[0] = errorsForUpstream0;
    errorsForUpstream[1] = errorsForUpstream1;
    ActivationBackprop *props[2];
    props[0] = p0;
    props[1] = p1;
    for( int it = 0; it < its; it++ ) {
        // selectors might go over the edge if we just choose random ints
        // easiest way to select valid selectors might be to just forwardpropagate first?

        WeightRandomizer::randomize( it, errors, errorsSize, -0.1f, 0.1f );
        WeightRandomizer::randomize( it, input, inputSize, -0.1f, 0.1f );    
        forwardprop->propagate( batchSize, input, output );

        for( int instance = 0; instance < 2; instance++ ) {
            props[instance]->backpropErrors( batchSize, input, errors, errorsForUpstream[instance] );
        }
        bool ok = true;
        int numErrors = 0;
        for( int i = 0; i < inputSize; i++ ) {
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
    delete[] errorsForUpstream0;
    delete[] errorsForUpstream1;
    delete p0;
    delete p1;
    delete cl;
}

/*
TEST( testactivationpropagate, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    int activationSize = 2;
    OpenCLHelper cl;
    ActivationPropagate *activationPropagate = ActivationPropagate::instanceForTest( cl, numPlanes, imageSize, activationSize );
    float data[] = { 1, 2, 
                    5, 3,

                     3, 8, 
                    4, 1,

                     3, 33, 
                    14,23,

                     -1, -3.5f,
                    37.4f,5
    };
    int outputSize = activationPropagate->getResultsSize( batchSize );
    int *selectors = new int[outputSize];
    float *output = new float[outputSize];

    activationPropagate->propagate( batchSize, data, selectors, output );

    EXPECT_EQ( selectors[0], 2 );
    EXPECT_EQ( selectors[1], 1 );
    EXPECT_EQ( selectors[2], 1 );
    EXPECT_EQ( selectors[3], 2 );

    EXPECT_EQ( output[0], 5 );
    EXPECT_EQ( output[1], 8 );
    EXPECT_EQ( output[2], 33 );
    EXPECT_EQ( output[3], 37.4f );

    delete activationPropagate;
    delete[] selectors;
    delete[] output;
}
*/

