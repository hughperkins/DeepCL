// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"
#include "net/NeuralNet.h"
#include "conv/Forward.h"
#include "activate/ActivationFunction.h"
#include "layer/Layer.h"
#include "layer/LayerMakers.h"
#include "util/StatefulTimer.h"
#include "net/NeuralNetMould.h"
#include "clblas/ClBlasInstance.h"
#include "clBLAS.h"

#include "test/WeightRandomizer.h"
#include "test/DeepCLGtestGlobals.h"
#include "test/TestArgsParser.h"
#include "test/DimFromArgs.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

#include "gtest/gtest.h"

using namespace std;

#include "test/gtest_supp.h"

void forwardWithWipe( Forward *prop, int batchSize, LayerDimensions dim, float *inputData, float *filters, float *biases, float *output ) {
    int inputDataSize = batchSize * dim.inputCubeSize;
    CLWrapper *dataWrapper = prop->cl->wrap( inputDataSize, inputData );
    dataWrapper->copyToDevice();

    int weightsSize = dim.filtersSize;
    CLWrapper *weightsWrapper = prop->cl->wrap( weightsSize, filters );
    weightsWrapper->copyToDevice();

    CLWrapper *biasWrapper = 0;
    if( dim.biased ) {
        biasWrapper = prop->cl->wrap( dim.numFilters, biases );
        biasWrapper->copyToDevice();
    }

    CLWrapper *outputWrapper = prop->cl->wrap( batchSize * dim.outputCubeSize, output );
    memset( output, 99, sizeof(float) * batchSize * dim.outputCubeSize );
    outputWrapper->copyToDevice(); // so we can wipe it...

    StatefulTimer::timeCheck("testforward: after data wrapper processing");
    prop->forward( batchSize, dataWrapper, weightsWrapper, biasWrapper,
            outputWrapper );
//    StatefulTimer::timeCheck("Forward::forward after call forward");
    outputWrapper->copyToHost();
//    StatefulTimer::timeCheck("Forward::forward after copytohost");
    delete outputWrapper;

    delete dataWrapper;
    delete weightsWrapper;
    if( dim.biased ) {
        delete biasWrapper;
    }
}

TEST( testforward, imagesize2_nopadzeros ) {
    int batchSize = 2;
    int numInPlanes = 1; int imageSize = 2;
    int numOutPlanes = 2; int filterWidth = 2;
    int padZeros = 0;
    float data[] = { 0, 0, 
                      0.5f, 0.5f,

                        13, 17,
                       -19, 2.3f,
};
    float filter1[] = { 0, 0,
                        -0.5f, 0.5f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,
 };
    int resultSize = 4;
    float expectedOutput[] = {
        -0.5f * 0.5f + 0.5f * 0.5f,
        0.7f * 0.5f -1.1f * 0.5f,
        (-0.5f) * (-19) + 0.5f * 2.3f,
        0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f 
    };
    cout << "expected number of output: " << resultSize << endl;
//    int outputSize = 0;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    for( int i = 1; i <= 4; i++ ) {
        Forward *forward = Forward::instanceSpecific( 3, cl,
            LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
            padZeros == 1, false ) );
        float *output = new float[forward->getOutputTotalSize(batchSize)];
        forward->forward( batchSize, data, filter1, 0, output );  
        for( int result = 0; result < resultSize; result++ ) {
            ASSERT_EQ( expectedOutput[result], output[result] );
        }
        delete forward;
        delete[] output;
    }

    delete cl;
}

TEST( testforward, DISABLED_imagesize2_nopadzeros_skip1 ) {
    int batchSize = 2;
    int numInPlanes = 1; int imageSize = 4;
    int numOutPlanes = 2; int filterWidth = 2;
    int padZeros = 0;
    int skip = 1;
    float data[] = { 0, 1, 3, 0, 
                    4, 0, 0, 0, 
                      0.5f, 0, 0.5f,0, 
                      0,    0, 0,   0, 

                        13, 0, 17,0, 
                        0, 0, 0, 0, 
                       -19, 0, 2.3f,0, 
                        0, 0, 0, 0, 
};
    float filter1[] = { 0, 0,
                        -0.5f, 0.5f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,
 };
    int outputSize = ( imageSize - filterWidth ) / ( skip + 1 ) + 1;
    cout << "outputimagesize: " << outputSize << endl;
    int outputNumElements = outputSize * numOutPlanes * batchSize;
    cout << "outputsize: " << outputNumElements << endl;
    float expectedOutput[] = {
        -2,  0,
        0, 0,

         2.8f, 0.6f,
         1.0f, 0.1f,

         0, 0,
         0,0,

         13*0.2f,17*0.2f,
         -19*0.2f, -2.3f*1.1f


    };
    cout << "expected number of output: " << outputNumElements << endl;
//    int outputSize = 0;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    for( int i = 1; i <= 1; i++ ) {
        Forward *forward = Forward::instanceSpecific( 0, cl,
            LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
            padZeros == 1, false ).setSkip(1) );
        float *output = new float[forward->getOutputTotalSize(batchSize)];
        forward->forward( batchSize, data, filter1, 0, output );  
        for( int result = 0; result < outputNumElements; result++ ) {
            cout << "checking result " << result << endl;
            EXPECT_EQ( expectedOutput[result], output[result] );
        }
        delete forward;
        delete[] output;
    }
    delete cl;
}

TEST( testforward, imagesize2_padzeros ) {
    int batchSize = 2;
    int numOutPlanes = 2;
    int numInPlanes = 1;
    int imageSize = 2;
    int filterWidth = 2;
    int padZeros = 1;

    float data[] = { 0, 0, 
                      0.5f, 0.3f,

                        13, 17,
                       -19, 2.3f,
};

    float filter1[] = { 0, 0,
                        -0.5f, 0.4f,

                        0.2f, 0.3f, 
                         0.7f, -1.1f,

 };
    int resultSize = (imageSize + 1) * (imageSize + 1) * batchSize * numOutPlanes;
    float *expectedOutput = new float[resultSize];
    for( int i = 0; i < resultSize; i++ ) {
        expectedOutput[i] = -9999; // means havent provided an expectedresult.
    }

    expectedOutput[0] = 0; expectedOutput[1] = 0; expectedOutput[2] = 0;

    expectedOutput[3] = 0.5f*0.4f;
    expectedOutput[4] = 0.5f*(-0.5f)+0.4f*(0.3f);
    expectedOutput[5] = 0.3f * (-0.5f); 

    expectedOutput[6] = 0; expectedOutput[7] = 0; expectedOutput[8] = 0;

    expectedOutput[9] = 0; expectedOutput[10] = 0; expectedOutput[11] = 0;
    expectedOutput[12] =(-1.1f)*0.5;
    expectedOutput[13] = 0.7f * 0.5f + (-1.1f) * 0.3f;
    expectedOutput[14] = 0.7f * 0.3f;

    // plane 2, filter 2 ...
    expectedOutput[27] = (-1.1f*13);
    expectedOutput[28] = 0.7f * 13 + (-1.1f)*17;
    expectedOutput[29] = 0.7f*17;
    expectedOutput[35] = 0.2f* 2.3f;

//    expectedOutput[] = 0;
//    expectedOutput[5] = 0;
//    expectedOutput[6] = 0.3f * 0.5f;
//    expectedOutput[7] = 0.2f * 0.5f;

//    expectedOutput[8] = 13 * 0.5f;
//    expectedOutput[9] = 17 * (-0.5f);
//    expectedOutput[10] = (-19) * 0;
//    expectedOutput[11] = 2.3f * 0;
// 
//    expectedOutput[12] = 13 * (-1.1f);
//    expectedOutput[13] = 17 * 0.7f;
//    expectedOutput[14] = (-19) * 0.3f;
//    expectedOutput[15] = 2.3f * 0.2f;

//        -0.5f * 0.5f + 0.5f * 0.5f,
//        0.7f * 0.5f -1.1f * 0.5f,
//        (-0.5f) * (-19) + 0.5f * 2.3f,
//        0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f 
//    };

//    int outputSize = 0;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    Forward *forward = Forward::instanceTest( cl, LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
        padZeros == 1, false ) );
    float *output = new float[forward->getOutputTotalSize(batchSize)];
    forward->forward( batchSize, data, filter1, 0, output );        

//    ASSERT_EQ( -0.5f * 0.5f + 0.5f * 0.5f, output[0] );
//    ASSERT_EQ( 0.7f * 0.5f -1.1f * 0.5f, output[1] );
//    ASSERT_EQ( (-0.5f) * (-19) + 0.5f * 2.3f, output[2] );
//    ASSERT_EQ( 0.2f*13 + 0.3f* 17 + 0.7f *(-19) -1.1f * 2.3f , output[3] );

    for( int result = 0; result < resultSize; result++ ) {
        if( expectedOutput[result] != -9999 ) {
            cout << " checking result[" << result << "]=" << output[result] << " expecting: " << expectedOutput[result] << endl;
            ASSERT_FLOAT_EQ( expectedOutput[result], output[result] );
        }
    }
    delete forward;
    delete[] output;
    delete cl;
}

TEST( testforward, imagesize3 ) {
    int batchSize = 5;
    int numOutPlanes = 2;
    int numInPlanes = 1;
    int imageSize = 3;
    int filterWidth = 3;
    int padZeros = 0;

    float data[] = { 0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, 0.5f,

                        0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, -0.5f ,

                        0, 0, 0,
                       0, 0, 0,
                       0.5f, 0, 0,

                        0, 0, 0,
                       0, 0, 0,
                       1, 10, 0,

                        0, 0, 0,
                       0, 0, 0,
                       0, 0, 1 
};

    float filter1[] = { 0, 0, 0,
                          0, 0, 0,
                         -0.5f, 0, 0.5f,

                        0, 0, 0,
                          0, 0, 0,
                         2.0f, 0.5, 0.5f,

 };

//    int outputSize = 0;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    Forward *forward = Forward::instanceTest( cl, LayerDimensions( numInPlanes, imageSize, numOutPlanes, filterWidth,
        padZeros == 1, false ) );
    float *output = new float[forward->getOutputTotalSize(batchSize)];
    forward->forward( 
        batchSize, data, filter1, 0, output );        

    EXPECT_EQ( 0, output[0] );
    EXPECT_EQ( 1.25f, output[1] );
    EXPECT_EQ( -0.5f, output[2] );
    EXPECT_EQ( 0.75f, output[3] );
    EXPECT_EQ( -0.25f, output[4] );
    EXPECT_EQ( 1, output[5] );
    EXPECT_EQ( -0.5f, output[6] );
    EXPECT_EQ( 7, output[7] );
    EXPECT_EQ( 0.5f, output[8] );
    EXPECT_EQ( 0.5f, output[9] );
        cout << "test1 ok" << endl;
    delete forward;
    delete[] output;
    delete cl;
}

TEST( testforward, test2 ) {
    int batchSize = 2;
    LayerDimensions dim;
    dim.setNumFilters(2).setNumInputPlanes(1).setInputSize(3).setFilterSize(3)
        .setPadZeros(false).setBiased(false);

    float data[] = { 0, 0, 0,
                       -0.5f, 0.5f, 0,
                       0, 0, 0,

                        0, 0, 0,
                       0.5f, -0.5f, 0,
                       0, 0, 0
};
    float filter1[] = { 0, 0, 0,
                          0.300809f, -0.11011f, 0,
                         0, 0, 0,

                        0, 0, 0,
                          0.0570846f, 0.347077f, 0,
                         0,0,0
 };

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float *biases = 0;

    Forward *forward = Forward::instanceSpecific( 1, cl, dim );
    float *output = new float[forward->getOutputTotalSize(batchSize)];
    forward->forward( batchSize, data, filter1, biases, output );

    EXPECT_FLOAT_NEAR( -0.5f * 0.300809f -0.5f * 0.11011f, output[0] );
    EXPECT_FLOAT_NEAR( -0.5f * 0.0570846f +0.5f * 0.347077f, output[1] );
    EXPECT_FLOAT_NEAR( 0.5f * 0.300809f +0.5f * 0.11011f, output[2] );
    EXPECT_FLOAT_NEAR( 0.5f * 0.0570846f -0.5f * 0.347077f, output[3] );

    delete[] output;
    delete forward;
    delete cl;
}

TEST( testforward, test3 ) {
    int batchSize = 4;
    int numInPlanes = 2;
    int numOutPlanes = 2;
    int inImageSize = 1;
//    int outImageSize = 1;
    int filterSize = 1;
    int padZeros = 0;
    float data[] = {0.1f,0.2f,
                    0.3f,0.4f,
                    0.5f,0.6f,
                    0.7f,0.8f};
    float filter[] = {0.2f,0.3f,
                     0.5f,0.7f};

//    int outputSize = 0;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    Forward *forward = Forward::instanceTest( cl, LayerDimensions( numInPlanes, inImageSize, numOutPlanes, filterSize,
        padZeros == 1, false ) );
    float *output = new float[forward->getOutputTotalSize(batchSize)];
    forward->forward( 
        batchSize, data, filter, 0, output );        

    float expectedOutput[] = {0.2f*0.1f+0.3f*0.2f,
                               0.5f*0.1f+0.7f*0.2f,

                               0.2f*0.3f+0.3f*0.4f,
                               0.5f*0.3f+0.7f*0.4f,

                                0.2f*0.5f+0.3f*0.6f,
                               0.5f*0.5f+0.7f*0.6f,
 
                              0.2f*0.7f+0.3f*0.8f,
                               0.5f*0.7f+0.7f*0.8f
  };
   for( int i = 0; i < 8; i++ ) {
//      cout << " checking result " << i << endl;
//        cout << "output[" << i << "]=" << output[i] << endl;
      EXPECT_FLOAT_NEAR( expectedOutput[i], output[i] );
   }
    delete[] output;
    delete forward;
    delete cl;
}

void compareSpecific( bool debug, int N, int batchSize, LayerDimensions dim, int instance0, int instance1 ) {
    cout << dim << endl;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance clblasInstance;

    int inputsSize = N * dim.inputCubeSize;
    int filtersSize = dim.filtersSize;
    int biasSize = dim.numFilters;
    int inputsAllocated = std::max( inputsSize, 0000 );
    int filtersAllocated = std::max( filtersSize, 0000 );
    int biasFiltersAllocated = std::max( biasSize, 0000 );
    float *inputs = new float[ inputsAllocated ];
    float *filters = new float[ filtersAllocated ];
    float *biasFilters = new float[ biasFiltersAllocated ];

//    memset( inputs, 0, sizeof(float) * inputsAllocated );
//    memset( filters, 0, sizeof(float) * filtersAllocated );
//    memset( biasFilters, 0, sizeof(float) * biasFiltersAllocated );

////    inputs[0] = 2.0f;
////    inputs[1] = 4.0f;
//    inputs[4] = 4.0f;
////    inputs[dim.inputB + 0] = 3.0f;
//    inputs[dim.inputCubeSize + 0] = 3.0f;

////    filters[0] = 3.0f;
////    filters[1] = 5.0f;
//    filters[4] = 5.0f;

    WeightRandomizer::randomize( 1, inputs, inputsAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( 2, filters, filtersAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( 3, biasFilters, biasFiltersAllocated, -0.1f, 0.1f );
    for( int i = 0; i < 8; i++ ) {
        if( debug ) cout << "i " << i << " input[i]=" << inputs[i] << " filters[i]=" << filters[i] << endl;
    }

    int outputNumElements = N * dim.outputCubeSize;
    float *output1 = new float[ outputNumElements ];
    float *output2 = new float[ outputNumElements ];
    
    int numBatches = ( N + batchSize - 1 ) / batchSize;
    Forward *p1 = Forward::instanceSpecific( instance0, cl, dim );
    Forward *p2 = Forward::instanceSpecific( instance1, cl, dim );

//    float *outputtemps[2];
    for( int instance = 0; instance < 2; instance++ ) {
        Forward *thisForward = 0;
        float *thisOutput = 0;
        if( instance == 0 ) { 
            thisForward = p1;
            thisOutput = output1;
        }
        if( instance == 1 ) {
            thisForward = p2;
            thisOutput = output2;
        }
        for( int batch = 0; batch < numBatches; batch++ ) {
            int thisBatchSize = batchSize;
            if( batch == numBatches - 1 ) {
                thisBatchSize = N - batch * batchSize;
            }
            cout << "batch " << batch << " batchsize " << thisBatchSize << endl;
            float *outputtemp = new float[thisBatchSize * dim.outputCubeSize * sizeof(float)];
//            memset( outputtemp, 123, thisBatchSize * dim.outputCubeSize * sizeof(float) ); // so kernel
                // cant just reuse the work of previous forward :-)
//            outputtemps[instance] = 
//            StatefulTimer::timeCheck("after memset");
            forwardWithWipe( thisForward, thisBatchSize, dim, inputs + batchSize * batch * dim.inputCubeSize, filters, biasFilters, outputtemp );
//            thisForward->forward( thisBatchSize, inputs + batchSize * batch * dim.inputCubeSize, filters, biasFilters, outputtemp );
            memcpy( thisOutput + batch * batchSize * dim.outputCubeSize, outputtemp, thisBatchSize * dim.outputCubeSize * sizeof(float) );
            delete[] outputtemp;
        }
        StatefulTimer::dump(true);
    }

    cout << dim << endl;
    bool same = true;
    int numDiff = 0;
    for( int i = 0; i < max( 20, outputNumElements ); i++ ) {
        if( i < outputNumElements ) {
            if( abs( output1[i] - output2[i] ) < 0.00001f || abs( output1[i] - output2[i] ) <= 0.001f * max( abs( output1[i] ), abs( output2[i] ) ) ) {
                if( i < 20 ) {
                    if( debug ) cout << "output[" << i << "]=" << output1[i] << " " << output2[i];
                    if( debug ) cout << " SAME";
                }
            } else {
                cout << "output[" << i << "]=" << output1[i] << " " << output2[i];
                cout << " DIFF";
                same = false;
                numDiff++;
            }
        } else {
             if( i < 20 ) {
                 if( debug ) cout << "     ";
             }
        }
        if( i < 20 ) {
            if( debug ) cout << "  || " << output2[100+i] ;
            if( debug ) cout << "  || " << output2[200+i] ;
            if( debug ) cout << "  || " << output2[300+i] ;
            if( debug ) cout << "  || " << output2[400+i] ;
            if( debug ) cout << "  || " << output2[500+i] ;
            if( debug ) cout << "  || " << output2[600+i] ;
            if( debug ) cout << "  || " << output2[700+i] << endl;
        }
        if( numDiff > 30 ) {
            cout << "..." << endl;
            break;
        }
    }
    EXPECT_EQ( true, same );
    delete[] output1;
    delete[] output2;
    delete p1;
    delete p2;
    delete[] inputs;
    delete[] filters;
    delete[] biasFilters;
    delete cl;
}

// first, compare the slow, but probably correct, cpu version, with forward1
// forward1 is slow-ish, but faster than cpu, and simple, so more likely to be correct
// then compare forward1 with each other type
TEST( testforward, compare_0_1_biased_nopad ) {
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputSize(19).setNumFilters( 8 )
        .setFilterSize( 5 )
        .setPadZeros( false ).setBiased( true );
    compareSpecific( false, N, batchSize, dim, 0, 1 );
}

TEST( testforward, compare_0_1_biased_pad ) {
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputSize(19).setNumFilters( 8 )
        .setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );
    compareSpecific( false, N, batchSize, dim, 0, 1 );
}

TEST( testforward, compare_1_n_biased_nopad ) {
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputSize(19).setNumFilters( 8 )
        .setFilterSize( 5 )
        .setPadZeros( false ).setBiased( true );
    for( int instance = 2; instance <= 7; instance++ ) {
        if( instance == 5 ) {
            continue; // forwardfc, cant use for inputimagesize != filtersize
        }
        cout << "instance: " << instance << endl;
        compareSpecific( false, N, batchSize, dim, 1, instance );
    }
}

TEST( testforward, compare_1_n_biased_pad ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    int maxWorkgroupSize = cl->getMaxWorkgroupSize();
    delete cl;

    LayerDimensions dim;
    int batchSize = 4;
    int N = 4;
    string activationName = "tanh";
    dim.setInputPlanes( 8 ).setInputSize(19).setNumFilters( 8 )
        .setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );
    for( int instance = 2; instance <= 7; instance++ ) {
        if( instance == 5 ) {
            continue; // forwardfc, cant use for inputimagesize != filtersize
        }
        dim.setInputSize(19);
        if(instance == 2 && maxWorkgroupSize < 19 * 19) {
            dim.setInputSize(15);
        }
        if(instance == 3 && maxWorkgroupSize < 19 * 19) {
            dim.setInputSize(15);
        }
        cout << "instance: " << instance << endl;
        compareSpecific( false, N, batchSize, dim, 1, instance );
    }
}

TEST( testforward, compare_1_5_biased_nopad ) { // only need to do nopad, since fc wont work with pad
    LayerDimensions dim;
    int batchSize = 4;
//    int instance0 = 1;
//    int instance1 = 1;
    int N = 4;
    dim.setInputPlanes( 8 ).setInputSize(19).setNumFilters( 8 )
        .setFilterSize( 19 )
        .setPadZeros( false ).setBiased( true );
    compareSpecific( false, N, batchSize, dim, 1, 5 );
}

TEST( testforward, compare_1_4_fcscenario ) { // only need to do nopad, since fc wont work with pad
    LayerDimensions dim;
    int batchSize = 4;
    int N = 4;
    dim.setInputPlanes( 10 ).setInputSize(24).setNumFilters( 10 )
        .setFilterSize( 24 )
        .setPadZeros( false ).setBiased( true );    
    compareSpecific( false, N, batchSize, dim, 1, 4 );
}

/* [[[cog
    for n in [1, 4]:
        cog.outl(
            'TEST( testforward, compare_break1_0_{n} ) {{\n'
            '    LayerDimensions dim;\n'
            '    dim.setInputPlanes( 1 ).setInputSize( 33 ).setNumFilters( 1 ).setFilterSize( 1 )\n'
            '        .setPadZeros( false ).setBiased( false );\n'
            '    compareSpecific( false, 1, 1, dim, 0, {n} );\n'
            '}}\n'.format(
                n=n))
*///]]]
TEST( testforward, compare_break1_0_1 ) {
    LayerDimensions dim;
    dim.setInputPlanes( 1 ).setInputSize( 33 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setPadZeros( false ).setBiased( false );
    compareSpecific( false, 1, 1, dim, 0, 1 );
}

TEST( testforward, compare_break1_0_4 ) {
    LayerDimensions dim;
    dim.setInputPlanes( 1 ).setInputSize( 33 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setPadZeros( false ).setBiased( false );
    compareSpecific( false, 1, 1, dim, 0, 4 );
}

// [[[end]]]

//TEST( SLOW_testforward, comparespecific ) {
//    LayerDimensions dim;
//    dim.setInputPlanes( 2 ).setInputSize(5).setNumFilters( 1 ).setFilterSize( 5 )
//        .setPadZeros( true ).setBiased( false );    
//    compareSpecific( 1, dim, 1, 3 );
//}

//TEST( SLOW_testforward, comparespecific_fc500unbiased ) {
//    LayerDimensions dim;
//    const int imageSize = 19;
//    dim.setInputPlanes( 32 ).setInputSize(imageSize).setNumFilters( 500 ).setFilterSize( imageSize )
//        .setPadZeros( false ).setBiased( false );    
//    compareSpecific( 4, dim, 1, 5 );
//}

//TEST( SLOW_testforward, comparespecific_fc500biased ) {
//    LayerDimensions dim;
//    const int imageSize = 19;
//    dim.setInputPlanes( 32 ).setInputSize(imageSize).setNumFilters( 500 ).setFilterSize( imageSize )
//        .setPadZeros( false ).setBiased( true );    
//    compareSpecific( 4, dim, 1, 5 );
//}

//TEST( SLOW_testforward, comparespecific_kgsgo_64c7 ) {
//    LayerDimensions dim;
//    const int imageSize = 19;
//    dim.setInputPlanes( 64 ).setInputSize(imageSize).setNumFilters( 64 ).setFilterSize( 7 )
//        .setPadZeros( true ).setBiased( true );    
//    compareSpecific( 128, dim, new ReluActivation(), 1, 6 );
//}

TEST( SLOW_testforward, compare_args ) {
    LayerDimensions dim;
    int batchSize = 128;
    int instance0 = 1;
    int instance1 = 3;
    int N = 128;
    bool debug = false;
    dim.setInputPlanes( 64 ).setInputSize(19).setNumFilters( 64 )
        .setFilterSize( 7 )
        .setPadZeros( true ).setBiased( false );    

    TestArgsParser::arg( "n", &N );
    DimFromArgs::arg( &dim );
    TestArgsParser::arg( "instance0", &instance0 );
    TestArgsParser::arg( "instance1", &instance1 );
    TestArgsParser::arg( "debug", &debug );
    TestArgsParser::arg( "batchsize", &batchSize );
    TestArgsParser::go();
    dim.deriveOthers();

    compareSpecific( debug, N, batchSize, dim, instance0, instance1 );
}

TEST( testforward, comparespecific_break2 ) { // this breaks on v5.7.0 for example
    LayerDimensions dim;
    int batchSize = 4;
    int instance0 = 1;
    int instance1 = 5;
    int N = 4;
    bool debug = false;
    dim.setInputPlanes( 64 ).setInputSize(19).setNumFilters( 64 )
        .setFilterSize( 19 )
        .setPadZeros( false ).setBiased( false );    

    TestArgsParser::arg( "n", &N );
    DimFromArgs::arg( &dim );
    TestArgsParser::arg( "instance0", &instance0 );
    TestArgsParser::arg( "instance1", &instance1 );
    TestArgsParser::arg( "debug", &debug );
    TestArgsParser::arg( "batchsize", &batchSize );
    TestArgsParser::go();
    dim.deriveOthers();

    compareSpecific( debug, N, batchSize, dim, instance0, instance1 );    
}

//TEST( SLOW_testforward, comparespecific_kgsgo_64c7mini ) {
//    LayerDimensions dim;
//    const int imageSize = 9;
//    dim.setInputPlanes( 4 ).setInputSize(imageSize).setNumFilters( 4 ).setFilterSize( 5 )
//        .setPadZeros( true ).setBiased( false );    
//    compareSpecific( 4, dim, new ReluActivation(), 1, 6 );
//}

TEST( testforward, softmax ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = NeuralNet::maker(cl)->imageSize(1)->planes(4)->instance();
    net->addLayer( SoftMaxMaker::instance() );
    net->setBatchSize( 1 );
    float *input = new float[net->getLayer(0)->getOutputPlanes()];
    input[0] = 0;
    input[1] = 1;
    input[2] = 3;
    input[3] = 2;
    net->forward( input );
    float const*output = net->getOutput();
    float sum = 0;
    for( int i = 0; i < net->getLayer(0)->getOutputPlanes(); i++ ) {
        cout << "output[" << i << "]=" << output[i] << endl;
        sum += output[i];
        EXPECT_LE( 0, output[i] );
        EXPECT_GE( 1, output[i] );
    }
    EXPECT_FLOAT_NEAR( 1.0f, sum );
    EXPECT_FLOAT_NEAR( (float)( exp(0.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), output[0] );
    EXPECT_FLOAT_NEAR( (float)( exp(1.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), output[1] );
    EXPECT_FLOAT_NEAR( (float)( exp(3.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), output[2] );
    EXPECT_FLOAT_NEAR( (float)( exp(2.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), output[3] );

    float *expected = new float[net->getLayer(0)->getOutputPlanes()];
    memset( expected, 0, sizeof(float) * net->getLayer(0)->getOutputPlanes() );
    expected[2] = 1;
    float loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(output[2]), loss );

    memset( expected, 0, sizeof(float) * net->getLayer(0)->getOutputPlanes() );
    expected[0] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(output[0]), loss );

    memset( expected, 0, sizeof(float) * net->getLayer(0)->getOutputPlanes() );
    expected[1] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(output[1]), loss );

    memset( expected, 0, sizeof(float) * net->getLayer(0)->getOutputPlanes() );
    expected[3] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(output[3]), loss );

    delete[] input;
    delete[] expected;
    delete net;
    delete cl;
}

TEST( testforward, softmax_byplane ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = NeuralNet::maker(cl)->imageSize(2)->planes(1)->instance();
    net->addLayer( SoftMaxMaker::instance()->perPlane() );
    net->setBatchSize( 1 );
    int imageSizeSquared = net->getLayer(0)->getOutputSize() * net->getLayer(0)->getOutputSize();
    float *input = new float[imageSizeSquared];
    input[0] = 0;
    input[1] = 1;
    input[2] = 3;
    input[3] = 2;
    net->forward( input );
    float const*output = net->getOutput();
    float sum = 0;
    for( int i = 0; i < imageSizeSquared; i++ ) {
        cout << "output[" << i << "]=" << output[i] << endl;
        sum += output[i];
        EXPECT_LE( 0, output[i] );
        EXPECT_GE( 1, output[i] );
    }
    EXPECT_FLOAT_NEAR( 1.0f, sum );
    EXPECT_FLOAT_NEAR( (float)( exp(0.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), output[0] );
    EXPECT_FLOAT_NEAR( (float)( exp(1.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), output[1] );
    EXPECT_FLOAT_NEAR( (float)( exp(3.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), output[2] );
    EXPECT_FLOAT_NEAR( (float)( exp(2.0f)/(exp(0.0f)+exp(1.0f)+exp(3.0f)+exp(2.0f)) ), output[3] );

    float *expected = new float[imageSizeSquared];
    memset( expected, 0, sizeof(float) * imageSizeSquared );
    expected[2] = 1;
    float loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(output[2]), loss );

    memset( expected, 0, sizeof(float) * imageSizeSquared );
    expected[0] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(output[0]), loss );

    memset( expected, 0, sizeof(float) * imageSizeSquared );
    expected[1] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(output[1]), loss );

    memset( expected, 0, sizeof(float) * imageSizeSquared );
    expected[3] = 1;
    loss = net->calcLoss( expected );
    cout << "loss " << loss << endl;
    EXPECT_LT( 0, loss );
    EXPECT_FLOAT_NEAR( - log(output[3]), loss );

    delete[] input;
    delete[] expected;
    delete net;
    delete cl;
}

void testPerf( int instance, int N, int batchSize, LayerDimensions dim ) {
    cout << dim.buildOptionsString() << endl;  

    int inputsSize = batchSize * dim.inputCubeSize;
    int filtersSize = dim.filtersSize;
    int biasSize = dim.numFilters;
    int inputsAllocated = std::max( inputsSize, 10000 );
    int filtersAllocated = std::max( filtersSize, 10000 );
    int biasFiltersAllocated = std::max( biasSize, 10000 );
    float *inputs = new float[ inputsAllocated ];
    float *filters = new float[ filtersAllocated ];
    float *biasFilters = new float[ biasFiltersAllocated ];

    memset( inputs, 0, sizeof(float) * inputsAllocated );
    memset( filters, 0, sizeof(float) * filtersAllocated );
    memset( biasFilters, 0, sizeof(float) * biasFiltersAllocated );

    WeightRandomizer::randomize( inputs, inputsAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( filters, filtersAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( biasFilters, biasFiltersAllocated, -0.1f, 0.1f );

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    Forward *p1 = Forward::instanceSpecific( instance, cl, dim );
    for( int it = 0; it < (N + batchSize - 1 ) / batchSize; it++ ) {
        int thisBatchSize = it < N - 1 ? batchSize : N - batchSize * it;
        float *output1 = new float[p1->getOutputTotalSize(thisBatchSize)];
        p1->forward( thisBatchSize, inputs, filters, biasFilters, output1 );
        delete[] output1;
    }
    StatefulTimer::dump(true);

    delete p1;
    delete cl;
    delete[] inputs;
    delete[] filters;
    delete[] biasFilters;
}

TEST( SLOW_testforward, perf_kgsgo_fc500 ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputSize(19).setNumFilters( 500 ).setFilterSize( 19 )
        .setPadZeros( false ).setBiased( true );  
    testPerf( -1, 128, batchSize, dim );
}

TEST( SLOW_testforward, perf_mnist_firstconvlayer ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 1 ).setInputSize(28).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    testPerf( -1, 128, batchSize, dim );
}

TEST( SLOW_testforward, perf_mnist_intlayers_128ex ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputSize(28).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    testPerf( -1, 128, batchSize, dim );
}

TEST( SLOW_testforward, perf_mnist_intlayers_1024ex ) {
    int batchSize = 1024;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputSize(28).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );    
    testPerf( -1, 128, batchSize, dim );
}

TEST( SLOW_testforward, perf_mnist_finallayer ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputSize(28).setNumFilters( 10 ).setFilterSize( 28 )
        .setPadZeros( false ).setBiased( true );    
    testPerf( -1, 128, batchSize, dim );
}

TEST( testforward, crash_from_jm ) {
    int instance = 1;
    int batchSize = 64;
    int N = 64;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputSize(28).setNumFilters( 20 ).setFilterSize( 28 )
        .setPadZeros( false ).setBiased( false );
    DimFromArgs::arg( &dim );
    TestArgsParser::arg( "instance", &instance );
    TestArgsParser::arg( "n", &N );
    TestArgsParser::arg( "batchsize", &batchSize );
    TestArgsParser::go();
    testPerf( instance, N, batchSize, dim );
}

TEST( SLOW_testforward, perf_kgsgo_64c7_args ) {
    int instance = 3;
    int batchSize = 128;
    int N = 1000;
    LayerDimensions dim;
    dim.setInputPlanes( 64 ).setInputSize(19).setNumFilters( 64 ).setFilterSize( 7 )
        .setPadZeros( true ).setBiased( true );
    DimFromArgs::arg( &dim );
    TestArgsParser::arg( "instance", &instance );
    TestArgsParser::arg( "n", &N );
    TestArgsParser::arg( "batchsize", &batchSize );
    TestArgsParser::go();
    testPerf( instance, N, batchSize, dim );
}

TEST( SLOW_testforward, soumith2 ) {
    int batchSize = 128;
    LayerDimensions dim;
    int instance = 4;
    bool biased = true;
    TestArgsParser::arg( "instance", &instance );
    TestArgsParser::arg( "biased", &biased );
    TestArgsParser::go();
    dim.setInputPlanes( 64 ).setInputSize( 64 ).setNumFilters( 128 ).setFilterSize( 9 )
        .setPadZeros( false ).setBiased( biased );  
    testPerf( instance, 128, batchSize, dim );
}

