// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "activate/ActivationBackward.h"
#include "activate/ActivationForward.h"
#include "activate/ActivationFunction.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/TestArgsParser.h"
#include "test/WeightRandomizer.h"

using namespace std;

TEST( testactivationbackward, basic ) {
    int batchSize = 1;
    int numPlanes = 1;
    int imageSize = 3;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ActivationBackward *activationBackprop = ActivationBackward::instanceForTest( cl, numPlanes, imageSize, new ReluActivation() );
    float outputs[] = {
        1, 0, 0.1f,
        0.5f, 0, 1000,
        2.5f, 2.0f, 0
    };
    float gradOutput[] = {
        3, 5,-2.7f,
        2, -9, 2.1f,
        0, -1.1f, 3.5f
    };
    int inputTotalSize = activationBackprop->getInputNumElements( batchSize );
    EXPECT_EQ( batchSize * imageSize * imageSize, inputTotalSize );
    float *gradInput = new float[ inputTotalSize ];

    activationBackprop->backward( batchSize, outputs, gradOutput, gradInput );

//    float *expectedGradInput = new float[ activationForward->getInputNumElements( batchSize ) ];
//    memset( expectedGradInput, 0, sizeof(float) * activationForward->getInputNumElements( batchSize ) ];
//    float expectedGradInput[] = {
//        3,0,-2.7f,
//        2,0,2.1f,
//        0,-1.1f,0,
//    };
    EXPECT_EQ( 3, gradInput[0] );
    EXPECT_EQ( 0, gradInput[1] );
    EXPECT_EQ( -2.7f, gradInput[2] );

    EXPECT_EQ( 2, gradInput[3] );
    EXPECT_EQ( 0, gradInput[4] );
    EXPECT_EQ( 2.1f, gradInput[5] );

    EXPECT_EQ( 0, gradInput[6] );
    EXPECT_EQ( -1.1f, gradInput[7] );
    EXPECT_EQ( 0, gradInput[8] );
//    for( int i = 0; i < 16; i++ ) {
//        EXPECT_EQ( expectedGradInput[i], gradInput[i] );
//    }

    delete activationBackprop;
    delete[] gradInput;
    delete cl;
}

TEST( testactivationbackward, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 1;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ActivationBackward *activationBackprop = ActivationBackward::instanceForTest( cl, numPlanes, imageSize, new ReluActivation() );
    float outputs[] = {
        2,
        0,
        0,
        2
    };
    float gradOutput[] = {
        3, 
        5,
        2, 
        9
    };
    float *gradInput = new float[ activationBackprop->getInputNumElements( batchSize ) ];

    activationBackprop->backward( batchSize, outputs, gradOutput, gradInput );

//    float *expectedGradInput = new float[ activationForward->getInputNumElements( batchSize ) ];
//    memset( expectedGradInput, 0, sizeof(float) * activationForward->getInputNumElements( batchSize ) ];
    float expectedGradInput[] = {
        3,
        0,
        0,
        9
    };
    for( int i = 0; i < 4; i++ ) {
        ASSERT_EQ( expectedGradInput[i], gradInput[i] );
    }

    delete activationBackprop;
    delete[] gradInput;
    delete cl;
}

TEST( SLOW_testactivationbackward, compare_args ) {
    int inputSize = 9;
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
    TestArgsParser::arg( "inputimagesize", &inputSize );
    TestArgsParser::arg( "instance0", &instance0 );
    TestArgsParser::arg( "instance1", &instance1 );
    TestArgsParser::go();

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ActivationBackward *p0 = ActivationBackward::instanceSpecific( instance0, cl, numPlanes, inputSize, ActivationFunction::fromName( activation ) );
    ActivationBackward *p1 = ActivationBackward::instanceSpecific( instance1, cl, numPlanes, inputSize, ActivationFunction::fromName( activation ) );
    int outputSize = p1->outputSize;
    int gradOutputNumElements = batchSize * outputSize * outputSize * numPlanes;
    float *gradOutput = new float[ gradOutputNumElements ];
    int inputNumElements = batchSize * inputSize * inputSize * numPlanes;
    float *gradInput0 = new float[ inputNumElements ];
    float *gradInput1 = new float[ inputNumElements ];
    
    ActivationForward *forwardprop = ActivationForward::instanceSpecific( 0, cl, numPlanes, inputSize, ActivationFunction::fromName( activation ) );
    float *output = new float[gradOutputNumElements];
    float *input = new float[inputNumElements];
    float *gradInput[2];
    gradInput[0] = gradInput0;
    gradInput[1] = gradInput1;
    ActivationBackward *props[2];
    props[0] = p0;
    props[1] = p1;
    for( int it = 0; it < its; it++ ) {
        // selectors might go over the edge if we just choose random ints
        // easiest way to select valid selectors might be to just forwardforward first?

        WeightRandomizer::randomize( it, gradOutput, gradOutputNumElements, -0.1f, 0.1f );
        WeightRandomizer::randomize( it, input, inputNumElements, -0.1f, 0.1f );    
        forwardprop->forward( batchSize, input, output );

        for( int instance = 0; instance < 2; instance++ ) {
            props[instance]->backward( batchSize, output, gradOutput, gradInput[instance] );
        }
        bool ok = true;
        int numErrors = 0;
        for( int i = 0; i < inputNumElements; i++ ) {
            if( gradInput0[i] != gradInput1[i] ) {
                cout << "diff: i=" << i << " " << gradInput0[i] << " != " << gradInput1[i] << endl;
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
            cout << " breaking after " << it << " its, because of FAIL gradOutput" << endl;
            break; // no point in continuing...
        }
    }

    delete forwardprop;
    delete[] input;
    delete[] output;
    delete[] gradOutput;
    delete[] gradInput0;
    delete[] gradInput1;
    delete p0;
    delete p1;
    delete cl;
}

/*
TEST( testactivationforward, basic_2plane_batchsize2 ) {
    int batchSize = 2;
    int numPlanes = 2;
    int imageSize = 2;
    int activationSize = 2;
    EasyCL cl;
    ActivationForward *activationForward = ActivationForward::instanceForTest( cl, numPlanes, imageSize, activationSize );
    float data[] = { 1, 2, 
                    5, 3,

                     3, 8, 
                    4, 1,

                     3, 33, 
                    14,23,

                     -1, -3.5f,
                    37.4f,5
    };
    int outputNumElements = activationForward->getOutputNumElements( batchSize );
    int *selectors = new int[outputNumElements];
    float *output = new float[outputNumElements];

    activationForward->forward( batchSize, data, selectors, output );

    EXPECT_EQ( selectors[0], 2 );
    EXPECT_EQ( selectors[1], 1 );
    EXPECT_EQ( selectors[2], 1 );
    EXPECT_EQ( selectors[3], 2 );

    EXPECT_EQ( output[0], 5 );
    EXPECT_EQ( output[1], 8 );
    EXPECT_EQ( output[2], 33 );
    EXPECT_EQ( output[3], 37.4f );

    delete activationForward;
    delete[] selectors;
    delete[] output;
}
*/

