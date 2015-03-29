// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <iomanip>
#include <algorithm>

#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "BackpropWeights2.h"
#include "BackpropWeights2Naive.h"

#include "test/myasserts.h"
#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"
#include "test/DimFromArgs.h"
#include "test/TestArgsParser.h"

using namespace std;

void test( int imageSize, int filterSize, int numPlanes, int batchSize ) {
    float learningRate = 0.01f;
//    const int batchSize = 1;
//    const int imageSize = 1;

    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(numPlanes)->imageSize(imageSize) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(1)->filterSize(filterSize)->biased(0)->tanh() );
    net->addLayer( SquareLossMaker::instance() );;
    net->setBatchSize( batchSize );

    int inputSize = net->layers[0]->getResultsSize();
    int resultsSize = net->layers[1]->getResultsSize();
    int weightsSize = net->layers[1]->getWeightsSize();

    float *inputData = new float[max(10000, inputSize )];
    float *expectedResults = new float[max(10000, resultsSize )];
    memset( inputData, 0, sizeof(float) * max(10000, inputSize ) );
    memset( expectedResults, 0, sizeof(float) * max(10000, resultsSize ) );
//    int seed = 0;
    std::mt19937 random = WeightRandomizer::randomize( inputData, max(10000, inputSize ), -1.0f, 1.0f );
    WeightRandomizer::randomize( random, expectedResults, max(10000, resultsSize ), -1.0f, 1.0f );
    WeightRandomizer::randomize( random, net->layers[1]->getWeights(), weightsSize, -0.1f, 0.1f );
    dynamic_cast<ConvolutionalLayer*>(net->layers[1])->weightsWrapper->copyToDevice();
//    for( int i = 0; i < inputSize; i++ ) {
//        cout << "inputData[" << i << "]=" << inputData[i] << endl;
//    }
//    for( int i = 0; i < resultsSize; i++ ) {
//        cout << "expectedResults[" << i << "]=" << expectedResults[i] << endl;
//    }

    float *weightsBefore = new float[weightsSize];
    float const*currentWeights = net->layers[1]->getWeights();
    for( int i = 0; i < weightsSize; i++ ) {
        weightsBefore[i] = currentWeights[i];
    }

//    net->print();
//    cout << "propagate" <<endl;
    net->propagate( inputData );
//    net->print();
    float loss = net->calcLoss(expectedResults);
//    float losslayer1 = dynamic_cast<LossLayer*>(net->layers[1])->calcLoss(expectedResults);
//    cout << "losslayer1 " << losslayer1 << endl;

//    cout << "backprop now" <<endl;
    net->print();
    net->backProp( learningRate, expectedResults );
//    net->layers[1]->print();
    net->propagate( inputData );
    net->print();
//    net->layers[1]->print();
    float loss2 = net->calcLoss(expectedResults);
    float lossChange = loss - loss2;
    cout << " loss " << loss << " loss2 " << loss2 << " change: " << lossChange << endl;

    dynamic_cast<ConvolutionalLayer*>(net->layers[1])->weightsWrapper->copyToHost();
    float const*newWeights = net->layers[1]->getWeights();
    float sumWeightDiff = 0;
    float sumWeightDiffSquared = 0;
    for( int i = 0; i < weightsSize; i++ ) {
        float diff = newWeights[i] - weightsBefore[i];
        sumWeightDiff += diff;
        sumWeightDiffSquared += diff * diff;
    }
    cout << "sumweightsdiff " << sumWeightDiff << endl;
//    cout << "sumweightsdiff / learningrate " << (sumWeightDiff / learningRate ) << endl;
//    cout << "sum weightsdiffsquared " << (sumWeightDiffSquared/ learningRate / learningRate * imageSize ) << endl;

    float estimatedLossChangeFromW = sumWeightDiffSquared/ learningRate; // / filterSize;

    cout << " loss change              " << lossChange << endl;
    cout << " estimatedLossChangeFromW " << estimatedLossChangeFromW << endl;
//    cout << abs(estimatedLossChangeFromW - lossChange ) / lossChange << endl;    
//    cout << abs(estimatedLossChangeFromW - lossChange ) / estimatedLossChangeFromW << endl;    
    EXPECT_GT( 0.01f * imageSize * imageSize, abs(estimatedLossChangeFromW - lossChange ) / lossChange ); 
    EXPECT_GT( 0.01f * imageSize * imageSize, abs(estimatedLossChangeFromW - lossChange ) / estimatedLossChangeFromW ); 

//    delete[] weights1;
//    delete[] errors;
//    delete[] results;
    delete[] inputData;
}

TEST( testbackpropweights, numericallytest ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(1, 1, 1, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(3, 1, 1, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize5 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 1, 1, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize9 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 1, 1, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize9_filtersize9 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 9, 1, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize9_filtersize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 3, 1, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize3_filtersize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(3, 3, 1, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize5_filtersize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 1, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize5_filtersize3_batchsize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 1, 3 );
}

TEST( testbackpropweights, numericallytest_imagesize5_filtersize3_planes3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 3, 1 );
}

TEST( testbackpropweights, numericallytest_imagesize5_filtersize3_planes3_batchsize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 3, 3 );
}

void testBackpropWeights( LayerDimensions &dim, int batchSize, float learningMultiplier, float *data, float *errors, float * expectedResults ) {
    float *results = new float[batchSize * dim.outputCubeSize]; // ignored, for LINEAR
    float *weights = new float[max(dim.filtersSize,20)];
    float *biasWeights = new float[10];
    memset( weights, 0, sizeof( float ) * max( dim.filtersSize, 20 ) );
    memset( biasWeights, 0, sizeof(float) * 10 );

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    BackpropWeights2 *backpropWeightsImpl = BackpropWeights2::instanceForTest( cl, dim );
    backpropWeightsImpl->backpropWeights( batchSize, learningMultiplier, errors, data, weights, biasWeights );
    delete backpropWeightsImpl;
    
//    for( int i = 0; i < 20; i++ ) {
//        cout << "weights[" << i << "]=" << weights[i] << endl;
//    }
    for( int i = 0; i < dim.filtersSize; i++ ) {
        if( expectedResults[i] != -999 && expectedResults[i] != weights[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weights[i] );
        }
    }
    delete[] results;
    delete[] weights;
    delete[] biasWeights;
    delete cl;
}

TEST( testbackpropweights, backprop_weights_2 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 1 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );

    const int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f };
    float errors[] = { 7.0f };
    float expectedResults[] = { - 3 * 7 };
    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}


TEST( testbackpropweights, backprop_weights_2_upstreamimagesize2 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 2 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f, 13,
                    17, 19 };
    float DerivLossBySum[] = { 7.0f, 2,
                       4,4 };
    float expectedResults[] = { -3 * 7 - 13 * 2 // -191
                                 -17*4 -19*4 };   // 

    testBackpropWeights( dim, batchSize, learningMultiplier, data, DerivLossBySum, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamimagesize3_filtersize3 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 3 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 3 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f, 13, 5,
                    17, 19, -3,
                    2, -4, 7 };
    float errors[] = { 7.0f };
    float expectedResults[] = { -7 * 3, - 7 * 13, - 7 * 5, // -21 -91, -35
                                -7 * 17, - 7 * 19, 7 * 3,   // -119, 133, 21
                                - 7 * 2,  7 * 4, - 7 * 7 }; // -14, 28, -49

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamimagesize4_filtersize3 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 4 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 3 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f, 13, 5, 8,
                    17, 19, -3, 2,
                    2, -4, 7, 0,
                    0, 6, 8, 9 };
    float errors[] = { 7.0f, 2,
                        0, -3 };
    float expectedResults[] = { -3*7-13*2-0+19*3, -999, -999 , // 10
                                -999, -999, -999,
                                -999, -999, -49+27 };          //           -22

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamimagesize5_filtersize3 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 5 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 3 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f, 13,  5, 8, 3,
                    17,    19, -3, 2, 1,
                    2,     -4,  7, 0, -2,
                    0,     6,   8, 9, 4,
                     1,   3,    5, 3, 8 };
    float errors[] = { 7.0f, 2,-1,
                        0, -3,1,
                        2,-1,0 };
    float expectedResults[] = { -(3*7+13*2-1*5+0*17-3*19-1*3+2*2+1*4+0*7), -999, -999 , // 10
                                -999, -(19*7-3*2-2*1+  0-3*7+0*1   +2*6-1*8+0), -999,
                                -999, -999, -(7*7+0+2*1   +0-3*9+1*4   +5*2-1*3+0) };          //           -22
    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

float *allocateInputCleared( int batchSize, LayerDimensions &dim ) {
    int inputSize = batchSize * dim.inputCubeSize;
    float *data = new float[ inputSize ];
    memset( data, 0, sizeof(float) * inputSize );
    return data;
}

float *allocateErrorsCleared( int batchSize, LayerDimensions &dim ) {
    int resultsSize = batchSize * dim.outputCubeSize;
    float *errors = new float[ resultsSize ];
    memset( errors, 0, sizeof(float) * resultsSize );
    return errors;
}

TEST( testbackpropweights, backprop_weights_2_upstreamimagesize3_filtersize1 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 3 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared( batchSize, dim );
    data[0] = 2;
    data[1 * dim.inputImageSize + 1] = 7;
    data[2 * dim.inputImageSize + 2] = 5;

    float *errors = allocateErrorsCleared( batchSize, dim );
    errors[0] = 5;
    errors[1 * dim.outputImageSize + 1] = 11;
    errors[2 * dim.outputImageSize + 2] = 3;

    float expectedResults[] = { -(2 * 5 +  5 * 3 + 7 * 11 ) };          //           

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamimagesize16_filtersize1 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 16 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared( batchSize, dim );
    data[0] = 2;
    data[15 * dim.inputImageSize + 15] = 5;

    float *errors = allocateErrorsCleared( batchSize, dim );
    errors[0] = 4;
    errors[15 * dim.outputImageSize + 15] = 3;

    float expectedResults[] = { -(2 * 4 +  3 * 5 ) };          //           

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamimagesize17_filtersize1 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 17 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;
    cout << dim << endl;

    float *data = allocateInputCleared( batchSize, dim );
    data[0] = 2;
    data[1] = 3.2f;
    data[2] = 1.234f;
    data[16 * dim.inputImageSize + 16] = 5;

    float *errors = allocateErrorsCleared( batchSize, dim );
    errors[0] = 4;
    errors[1] = -2.5f;
    errors[2] = 4.125f;
    errors[16 * dim.outputImageSize + 16] = 3;

    float expectedResults[] = { -( 4*2 - 3.2f * 2.5f + 1.234f * 4.125f + 3*5 ) };          // 

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamimagesize17_filtersize1_moredata ) {
    LayerDimensions dim;
    dim.setInputImageSize( 17 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared( batchSize, dim );
    for( int i = 0; i < square( dim.inputImageSize ); i++ ) {
        data[i] = ( ( 1 + i ) % 20 ) / 5.3f;
    }

    float *errors = allocateErrorsCleared( batchSize, dim );
    for( int i = 0; i < square( dim.outputImageSize ); i++ ) {
        errors[i] = ( ( 2 + i ) % 17 ) / 4.2f;
    }

    float expectedResults[1];
    expectedResults[0] = 0;
    for ( int i = 0; i < square( dim.inputImageSize ); i++ ) {
        expectedResults[0] += - data[i] * errors[i];
    }
    cout << "expectedresult: " << expectedResults[0] << endl;

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_instance3_smaller2 ) {
    LayerDimensions dim;
    dim.setInputImageSize( 96 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 6 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningRate = 1;

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    int resultsSize = batchSize * dim.outputCubeSize;
    int inputSize = batchSize * dim.inputCubeSize;
    int weightsSize = dim.filtersSize;
//    int biasWeightsSize = dim.numFilters;

    cout << "numweights: " << weightsSize << endl;

    float *errors = new float[max(10000, resultsSize )];
    float *inputData = new float[max(10000, inputSize )];
    float *weights0 = new float[max(10000, weightsSize ) ];
    float *weights1 = new float[max(10000, weightsSize ) ];

    memset( errors, 0, sizeof(float) * max(10000, resultsSize ) );
    memset( inputData, 0, sizeof(float) * max(10000, inputSize ) );
    memset( weights0, 0, sizeof(float) * max(10000, weightsSize ) );
    memset( weights1, 0, sizeof(float) * max(10000, weightsSize ) );

    CLWrapper *errorsWrap = cl->wrap( 10000, errors );
    CLWrapper *inputWrap = cl->wrap( 10000, inputData );
    CLWrapper *weights0Wrap = cl->wrap( 10000, weights0 );
    CLWrapper *weights1Wrap = cl->wrap( 10000, weights1 );

    for( int i = 0 * dim.inputImageSize; i < dim.inputImageSize * dim.inputImageSize; i+= dim.inputImageSize * 4 ) {
        inputData[i] = 3;
    }
//    inputData[ 86 * 96 ] = 3;

//    inputData[ 0 ] = 3;

//    inputData[47 * 96] = 9;
//    inputData[48 * 96] = 3;

//    inputData[71 * 96] = 17;
//    inputData[72 * 96] = 13;

//    inputData[82 * 96] = 16;
//    inputData[83 * 96] = 18;
//    inputData[84 * 96] = 100;
//    inputData[85 * 96] = 42;
//    inputData[95 * 96] = 7;

    for( int i = 0; i < dim.outputImageSize * dim.outputImageSize; i+= dim.outputImageSize ) {
        errors[i] = 2;
    }

//    errors[0] = 4;

//    errors[46 * 93] = 4;
//    errors[47 * 93] = 6;

//    errors[81 * 93] = 4;
//    errors[82 * 93] = 15;
//    errors[83 * 93] = 8;

//    errors[84 * 93] = 3;
//    errors[85 * 93] = 9;

//    errors[92 * 93] = 5;


    errorsWrap->copyToDevice();
    inputWrap->copyToDevice();
    weights0Wrap->copyToDevice();
    weights1Wrap->copyToDevice();
    
    BackpropWeights2 *backpropWeightsImpl0 = BackpropWeights2::instanceSpecific( 0, cl, dim );
    backpropWeightsImpl0->debug = true;
    backpropWeightsImpl0->backpropWeights( batchSize, learningRate,
        errorsWrap, inputWrap, weights0Wrap, 0 );
    BackpropWeights2 *backpropWeightsImpl1 = BackpropWeights2::instanceSpecific( 3, cl, dim );
    backpropWeightsImpl1->debug = true;
    backpropWeightsImpl1->backpropWeights( batchSize, learningRate,
        errorsWrap, inputWrap, weights1Wrap, 0 );
    weights0Wrap->copyToHost();
    weights1Wrap->copyToHost();

    for( int i = 0; i < 6; i++ ) {
        for( int j = 0; j < 6; j++ ) {
            cout << weights0[i*6+j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    for( int i = 0; i < 6; i++ ) {
        for( int j = 0; j < 6; j++ ) {
            cout << weights1[i*6+j] << " ";
        }
        cout << endl;
    }

    cout << endl;
    int isok = 1;
    for( int i = 0; i < 6; i++ ) {
        for( int j = 0; j < 6; j++ ) {
            if( weights0[i*6+j] == weights1[i*6+j] ) {
                cout << ".";
            } else {
                cout << "!";
                isok = 0;
            }
        }
        cout << endl;
    }
    cout << endl;
    EXPECT_EQ( 1, isok );

    for( int i = 0; i < 12; i++ ) {
        cout << i << "=";
        for( int slice = 0; slice < 8; slice++ ) {
            cout << weights1[100+ 12 * slice + i] << " ";
        }
        cout << endl;
    }
    cout << endl;

    for( int i = 0; i < 20; i++ ) {
        cout << i << "=";
        for( int slice = 0; slice < 8; slice++ ) {
            cout << weights1[200+ 20 * slice + i] << " ";
        }
        cout << endl;
    }
    delete cl;
}

class CompareSpecificArgs {
public:
    static CompareSpecificArgs instance(){ CompareSpecificArgs args; return args; };

    // [[[cog
    // floats= []
    // ints = [  'inputPlanes', 'inputImageSize', 'numFilters', 'filterSize',
    //    'batchSize', 'biased', 'padZeros', 'instance0', 'instance1' ]
    // import cog_fluent
    // cog_fluent.gov2( 'CompareSpecificArgs', ints = ints, floats = floats )
    // ]]]
    // generated, using cog:
    int _inputPlanes = 0;
    int _inputImageSize = 0;
    int _numFilters = 0;
    int _filterSize = 0;
    int _batchSize = 0;
    int _biased = 0;
    int _padZeros = 0;
    int _instance0 = 0;
    int _instance1 = 0;
    CompareSpecificArgs inputPlanes( int _inputPlanes ) {
        this->_inputPlanes = _inputPlanes;
        return *this;
    }
    CompareSpecificArgs inputImageSize( int _inputImageSize ) {
        this->_inputImageSize = _inputImageSize;
        return *this;
    }
    CompareSpecificArgs numFilters( int _numFilters ) {
        this->_numFilters = _numFilters;
        return *this;
    }
    CompareSpecificArgs filterSize( int _filterSize ) {
        this->_filterSize = _filterSize;
        return *this;
    }
    CompareSpecificArgs batchSize( int _batchSize ) {
        this->_batchSize = _batchSize;
        return *this;
    }
    CompareSpecificArgs biased( int _biased ) {
        this->_biased = _biased;
        return *this;
    }
    CompareSpecificArgs padZeros( int _padZeros ) {
        this->_padZeros = _padZeros;
        return *this;
    }
    CompareSpecificArgs instance0( int _instance0 ) {
        this->_instance0 = _instance0;
        return *this;
    }
    CompareSpecificArgs instance1( int _instance1 ) {
        this->_instance1 = _instance1;
        return *this;
    }
    // [[[end]]]
};

namespace testbackpropweights {

void compareSpecific( bool debug, float learningRate, int its, int batchSize, LayerDimensions dim, int instance0, int instance1 ) {
    cout << dim << endl;

    int resultsSize = batchSize * dim.outputCubeSize;
    int inputSize = batchSize * dim.inputCubeSize;
    int weightsSize = dim.filtersSize;
    int biasWeightsSize = dim.numFilters;

    int resultsAllocated = max( 10000, resultsSize );
    int inputAllocated = max( 10000, inputSize );
    int weightsAllocated = max( 10000, weightsSize );
    int biasWeightsAllocated = max( 10000, biasWeightsSize );

//    cout << "numweights: " << weightsSize << endl;

    float *biasWeights1 = new float[ biasWeightsAllocated ];
    float *biasWeights2 = new float[ biasWeightsAllocated ];
    memset( biasWeights1, 0, sizeof(float) * biasWeightsAllocated );
    memset( biasWeights2, 0, sizeof(float) * biasWeightsAllocated );

    float *errors = new float[resultsAllocated];
    float *inputData = new float[inputAllocated];
    float *weights1 = new float[weightsAllocated];
    float *weights2 = new float[weightsAllocated];

    memset( errors, 0, sizeof(float) * resultsAllocated );
    memset( inputData, 0, sizeof(float) * inputAllocated );
    memset( weights1, 0, sizeof(float) * weightsAllocated );
    memset( weights2, 0, sizeof(float) * weightsAllocated );

    WeightRandomizer::randomize( errors, resultsAllocated, -0.1f, 0.1f );
    WeightRandomizer::randomize( inputData, inputAllocated, -0.3f, 0.7f );

//    WeightRandomizer::randomizeInts( errors, resultsAllocated, 0, 99 );
//    WeightRandomizer::randomizeInts( inputData, inputAllocated, 0, 99 );

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    
    int instances[2];
    instances[0] = instance0;
    instances[1] = instance1;
    float *weightsByInstance[2];
    weightsByInstance[0] = weights1;
    weightsByInstance[1] = weights2;
    float *biasWeightsByInstance[2];
    biasWeightsByInstance[0] = biasWeights1;
    biasWeightsByInstance[1] = biasWeights2;
    BackpropWeights2 *instanceObjects[2];
    instanceObjects[0] = BackpropWeights2::instanceSpecific( instance0, cl, dim );
    instanceObjects[1] = BackpropWeights2::instanceSpecific( instance1, cl, dim );
    for( int instance = 0; instance < 2; instance++ ) {
        Timer timer;
        BackpropWeights2 *backpropWeightsImpl = instanceObjects[instance];
        backpropWeightsImpl->debug = true;
        for( int it = 0; it < its; it++ ) {
            backpropWeightsImpl->backpropWeights( batchSize, learningRate,
                errors, inputData, weightsByInstance[instance], biasWeightsByInstance[instance] );
        }
        timer.timeCheck("instance " + toString( instances[instance] ) + " backpropweights" );
//        delete backpropWeightsImpl;
    }
    delete instanceObjects[0];
    delete instanceObjects[1];
    cout << dim << endl;
    for( int i = 0; i < 25; i++ ) {
        cout << "weights[" << i << "]=" << weights1[i] << " " << weights2[i];
        if( i < weightsSize ) {
            if( abs( weights1[i] - weights2[i] ) <= abs(weights1[i]) / 10000.0f ) {
                if( debug ) cout << " SAME";
            } else {
                cout << " DIFF";
            }
        } else {
            if( debug ) cout << "     ";
        }
        if( debug ) cout << "  || " << weights2[100+i] ;
        if( debug ) cout << "  || " << weights2[200+i] ;
        if( debug ) cout << "  || " << weights2[300+i] ;
        if( debug ) cout << "  || " << weights2[400+i] ;
        if( debug ) cout << "  || " << weights2[500+i] ;
        if( debug ) cout << "  || " << weights2[600+i] ;
        if( debug ) cout << "  || " << weights2[700+i] << endl;
    }
    bool same = true;
    int errCount = 0;
    for( int i = 0; i < weightsSize; i++ ) {
        if( abs( weights1[i] - weights2[i] ) > 0.001 * max( abs( weights1[i] ), abs( weights2[i] ) ) ) {
//        if( abs( weights1[i] - weights2[i] ) > abs(weights1[i]) / 10000.0f ) {
            cout << "DIFF: weights i " << i << " " << weights1[i] << " != " << weights2[i] << endl;
            same = false;
            errCount++;
            if( errCount == 5 ) {
                cout << " ... " << endl;
                break;
            }
        }
    }
    if( dim.biased ) {
        errCount = 0;
        for( int i = 0; i < biasWeightsSize; i++ ) {
            if( abs( biasWeights1[i] - biasWeights2[i] ) > 0.001 * max( abs( biasWeights1[i] ), abs( biasWeights2[i] ) ) ) {
    //        if( abs( weights1[i] - weights2[i] ) > abs(weights1[i]) / 10000.0f ) {
                cout << "DIFF: biasWeights i " << i << " " << biasWeights1[i] << " != " << biasWeights2[i] << endl;
                same = false;
                errCount++;
                if( errCount == 5 ) {
                    cout << " etc ... " << endl;
                    break;
                }
            }
        }
    }
    EXPECT_EQ( true, same );

//    delete backpropWeightsImpl1;
//    delete backpropWeightsImpl2;

    delete[] weights1;
    delete[] weights2;
    delete[] errors;
    delete[] inputData;

    delete cl;
}

TEST( SLOW_testbackpropweights, compare_args ) {
    bool debug = false;
    int instance0 = 1;
    int instance1 = 3;
    LayerDimensions dim;
    dim.setInputImageSize( 28 ).setInputPlanes( 1 ).setNumFilters( 8 ).setFilterSize( 5 )
        .setBiased( 1 ).setPadZeros( 1 );
    int batchSize = 4;
    int its = 1;
//        string activationName = "tanh";
    float learningRate = 1.0f;

    DimFromArgs::arg( &dim );
    TestArgsParser::arg( "debug", &debug );
    TestArgsParser::arg( "instance0", &instance0 );
    TestArgsParser::arg( "instance1", &instance1 );
    TestArgsParser::arg( "its", &its );
    TestArgsParser::arg( "batchsize", &batchSize );
//        TestArgsParser::arg( "activation", &activationName );
    TestArgsParser::arg( "learningrate", &learningRate );
    TestArgsParser::go();
    dim.deriveOthers();
//        ActivationFunction *fn = ActivationFunction::fromName( activationName );

    compareSpecific( debug, learningRate, its, batchSize, dim, instance0, instance1 );        
}

//    TEST( testbackpropweights, compare_instance3_smaller2 ) {
//        LayerDimensions dim;
//        dim.setInputImageSize( 96 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 6 )
//            .setBiased( 0 ).setPadZeros( 0 );
//        int batchSize = 1;
//        const float learningRate = 1;
//        compareSpecific( CompareSpecificArgs::instance()
//            .batchSize( 1 ).inputPlanes( 1 ).inputImageSize( 96 ).numFilters( 1 )
//            .filterSize( 6 ).biased( 0 ).padZeros( false )
//            .instance0(0).instance1(3) );
//    }

//    TEST( SLOW_testbackpropweights, compare_specific ) {
//        compareSpecific( CompareSpecificArgs::instance()
//            .batchSize( 128 ).inputPlanes( 32 ).inputImageSize( 19 ).numFilters( 32 )
//            .filterSize( 3 ).biased( 0 ).padZeros( false )
//            .instance0(1).instance1(3) );
//    }

//    TEST( SLOW_testbackpropweights, compare_specific_96image ) {
//        compareSpecific( CompareSpecificArgs::instance()
//            .batchSize( 128 ).inputPlanes( 2 ).inputImageSize( 96 ).numFilters( 8 )
//            .filterSize( 6 ).biased( 1 ).padZeros( false )
//            .instance0(0).instance1(3) );
//    }

//    TEST( SLOW_testbackpropweights, compare_specific_96image_smaller ) {
//        compareSpecific( CompareSpecificArgs::instance()
//            .batchSize( 1 ).inputPlanes( 1 ).inputImageSize( 48 ).numFilters( 1 )
//            .filterSize( 2 ).biased( 1 ).padZeros( false )
//            .instance0(0).instance1(3) );
//    }

//    TEST( SLOW_testbackpropweights, compare_specific_96image_smaller2 ) {
//        compareSpecific( CompareSpecificArgs::instance()
//            .batchSize( 1 ).inputPlanes( 1 ).inputImageSize( 96 ).numFilters( 1 )
//            .filterSize( 4 ).biased( 0 ).padZeros( false )
//            .instance0(0).instance1(3) );
//    }

//    TEST( SLOW_testbackpropweights, compare_specific_96image_smaller3 ) {
//        compareSpecific( CompareSpecificArgs::instance()
//            .batchSize( 1 ).inputPlanes( 1 ).inputImageSize( 96 ).numFilters( 1 )
//            .filterSize( 6 ).biased( false ).padZeros( false )
//            .instance0(0).instance1(3) );
//    }

//    TEST( SLOW_testbackpropweights, compare_specific_96image_smaller4 ) {
//        compareSpecific( CompareSpecificArgs::instance()
//            .batchSize( 1 ).inputPlanes( 2 ).inputImageSize( 96 ).numFilters( 8 )
//            .filterSize( 4 ).biased( 1 ).padZeros( false )
//            .instance0(0).instance1(3) );
//    }

void measurePerf( int batchSize, LayerDimensions dim, int instance ) {

    int resultsSize = batchSize * dim.outputCubeSize;
    int inputSize = batchSize * dim.inputCubeSize;
    int weightsSize = dim.filtersSize;
    int biasWeightsSize = dim.numFilters;

    int resultsAllocated = resultsSize;
    int inputAllocated = inputSize;
    int weightsAllocated = weightsSize;
    int biasWeightsAllocated = biasWeightsSize;

    cout << "numweights: " << weightsSize << endl;

    float *biasWeights = new float[ biasWeightsAllocated ];
    memset( biasWeights, 0, sizeof(float) * biasWeightsAllocated );

    float *errors = new float[resultsAllocated];
    float *inputData = new float[inputAllocated];
    float *weights = new float[weightsAllocated];

    memset( errors, 0, sizeof(float) * resultsAllocated );
    memset( inputData, 0, sizeof(float) * inputAllocated );
    memset( weights, 0, sizeof(float) * weightsAllocated );

    WeightRandomizer::randomizeInts( errors, resultsAllocated, 0, 99 );
    WeightRandomizer::randomizeInts( inputData, inputAllocated, 0, 99 );

    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    
    BackpropWeights2 *backpropWeightsImpl = BackpropWeights2::instanceSpecific( instance, cl, dim );
    Timer timer;
    backpropWeightsImpl->backpropWeights( batchSize, 1.0f,
        errors, inputData, weights, biasWeights );
    timer.timeCheck("backprop time");

    delete backpropWeightsImpl;

    delete[] weights;
    delete[] errors;
    delete[] inputData;

    delete cl;
}

}

