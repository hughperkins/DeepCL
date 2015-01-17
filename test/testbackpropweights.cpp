#include <iostream>
#include <iomanip>

#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "BackpropWeights2.h"

#include "test/myasserts.h"
#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"

using namespace std;

void test( int boardSize, int filterSize, int numPlanes, int batchSize ) {
    float learningRate = 0.01f;
//    const int batchSize = 1;
//    const int boardSize = 1;

    NeuralNet *net = NeuralNet::maker()->planes(numPlanes)->boardSize(boardSize)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(filterSize)->biased(0)->tanh()->insert();
    net->squareLossMaker()->insert();
    net->setBatchSize( batchSize );

    int inputSize = net->layers[0]->getResultsSize();
    int resultsSize = net->layers[1]->getResultsSize();
    int weightsSize = net->layers[1]->getWeightsSize();

    float *inputData = new float[max(10000, inputSize )];
    float *expectedResults = new float[max(10000, resultsSize )];
    memset( inputData, 0, sizeof(float) * max(10000, inputSize ) );
    memset( expectedResults, 0, sizeof(float) * max(10000, resultsSize ) );
    int seed = 0;
    std::mt19937 random = WeightRandomizer::randomize( inputData, max(10000, inputSize ), -1.0f, 1.0f );
    WeightRandomizer::randomize( random, expectedResults, max(10000, resultsSize ), -1.0f, 1.0f );
    WeightRandomizer::randomize( random, net->layers[1]->getWeights(), weightsSize, -0.1f, 0.1f );
    dynamic_cast<ConvolutionalLayer*>(net->layers[1])->weightsWrapper->copyToDevice();
    for( int i = 0; i < inputSize; i++ ) {
        cout << "inputData[" << i << "]=" << inputData[i] << endl;
    }
    for( int i = 0; i < resultsSize; i++ ) {
        cout << "expectedResults[" << i << "]=" << expectedResults[i] << endl;
    }

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
//    cout << "sum weightsdiffsquared " << (sumWeightDiffSquared/ learningRate / learningRate * boardSize ) << endl;

    float estimatedLossChangeFromW = sumWeightDiffSquared/ learningRate; // / filterSize;

    cout << " loss change              " << lossChange << endl;
    cout << " estimatedLossChangeFromW " << estimatedLossChangeFromW << endl;
//    cout << abs(estimatedLossChangeFromW - lossChange ) / lossChange << endl;    
//    cout << abs(estimatedLossChangeFromW - lossChange ) / estimatedLossChangeFromW << endl;    
    EXPECT_GT( 0.01f * boardSize * boardSize, abs(estimatedLossChangeFromW - lossChange ) / lossChange ); 
    EXPECT_GT( 0.01f * boardSize * boardSize, abs(estimatedLossChangeFromW - lossChange ) / estimatedLossChangeFromW ); 

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

TEST( testbackpropweights, numericallytest_boardsize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(3, 1, 1, 1 );
}

TEST( testbackpropweights, numericallytest_boardsize5 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 1, 1, 1 );
}

TEST( testbackpropweights, numericallytest_boardsize9 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 1, 1, 1 );
}

TEST( testbackpropweights, numericallytest_boardsize9_filtersize9 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 9, 1, 1 );
}

TEST( testbackpropweights, numericallytest_boardsize9_filtersize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 3, 1, 1 );
}

TEST( testbackpropweights, numericallytest_boardsize3_filtersize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(3, 3, 1, 1 );
}

TEST( testbackpropweights, numericallytest_boardsize5_filtersize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 1, 1 );
}

TEST( testbackpropweights, numericallytest_boardsize5_filtersize3_batchsize3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 1, 3 );
}

TEST( testbackpropweights, numericallytest_boardsize5_filtersize3_planes3 ) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 3, 1 );
}

TEST( testbackpropweights, numericallytest_boardsize5_filtersize3_planes3_batchsize3 ) {
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

    OpenCLHelper cl;
    BackpropWeights2 *backpropWeightsImpl = BackpropWeights2::instanceForTest( &cl, dim );
    backpropWeightsImpl->backpropWeights( batchSize, learningMultiplier, errors, data, weights, biasWeights );
    delete backpropWeightsImpl;
    
    for( int i = 0; i < 20; i++ ) {
        cout << "weights[" << i << "]=" << weights[i] << endl;
    }
    for( int i = 0; i < dim.filtersSize; i++ ) {
        if( expectedResults[i] != -999 && expectedResults[i] != weights[i] ) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ( expectedResults[i], weights[i] );
        }
    }
    delete[] results;
    delete[] weights;
    delete[] biasWeights;
}

TEST( testbackpropweights, backprop_weights_2 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 1 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );

    const int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f };
    float errors[] = { 7.0f };
    float expectedResults[] = { - 3 * 7 };
    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}


TEST( testbackpropweights, backprop_weights_2_upstreamboardsize2 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 2 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
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

TEST( testbackpropweights, backprop_weights_2_upstreamboardsize3_filtersize3 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 3 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 3 )
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

TEST( testbackpropweights, backprop_weights_2_upstreamboardsize4_filtersize3 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 4 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 3 )
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

TEST( testbackpropweights, backprop_weights_2_upstreamboardsize5_filtersize3 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 5 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 3 )
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

TEST( testbackpropweights, backprop_weights_2_upstreamboardsize3_filtersize1 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 3 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared( batchSize, dim );
    data[0] = 2;
    data[1 * dim.inputBoardSize + 1] = 7;
    data[2 * dim.inputBoardSize + 2] = 5;

    float *errors = allocateErrorsCleared( batchSize, dim );
    errors[0] = 5;
    errors[1 * dim.outputBoardSize + 1] = 11;
    errors[2 * dim.outputBoardSize + 2] = 3;

    float expectedResults[] = { -(2 * 5 +  5 * 3 + 7 * 11 ) };          //           

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamboardsize16_filtersize1 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 16 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared( batchSize, dim );
    data[0] = 2;
    data[15 * dim.inputBoardSize + 15] = 5;

    float *errors = allocateErrorsCleared( batchSize, dim );
    errors[0] = 4;
    errors[15 * dim.outputBoardSize + 15] = 3;

    float expectedResults[] = { -(2 * 4 +  3 * 5 ) };          //           

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamboardsize17_filtersize1 ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 17 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;
    cout << dim << endl;

    float *data = allocateInputCleared( batchSize, dim );
    data[0] = 2;
    data[1] = 3.2f;
    data[2] = 1.234f;
    data[16 * dim.inputBoardSize + 16] = 5;

    float *errors = allocateErrorsCleared( batchSize, dim );
    errors[0] = 4;
    errors[1] = -2.5f;
    errors[2] = 4.125f;
    errors[16 * dim.outputBoardSize + 16] = 3;

    float expectedResults[] = { -( 4*2 - 3.2f * 2.5f + 1.234f * 4.125f + 3*5 ) };          // 

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( testbackpropweights, backprop_weights_2_upstreamboardsize17_filtersize1_moredata ) {
    LayerDimensions dim;
    dim.setInputBoardSize( 17 ).setInputPlanes( 1 ).setNumFilters( 1 ).setFilterSize( 1 )
        .setBiased( 0 ).setPadZeros( 0 );
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared( batchSize, dim );
    for( int i = 0; i < square( dim.inputBoardSize ); i++ ) {
        data[i] = ( ( 1 + i ) % 20 ) / 5.3f;
    }

    float *errors = allocateErrorsCleared( batchSize, dim );
    for( int i = 0; i < square( dim.outputBoardSize ); i++ ) {
        errors[i] = ( ( 2 + i ) % 17 ) / 4.2f;
    }

    float expectedResults[1];
    expectedResults[0] = 0;
    for ( int i = 0; i < square( dim.inputBoardSize ); i++ ) {
        expectedResults[0] += - data[i] * errors[i];
    }
    cout << "expectedresult: " << expectedResults[0] << endl;

    testBackpropWeights( dim, batchSize, learningMultiplier, data, errors, expectedResults );
}

TEST( SLOW_testbackpropweights, compare_specific ) {
    const int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputBoardSize( 19 ).setNumFilters( 32 ).setFilterSize( 3 )
        .setBiased( false ).setPadZeros( true );
//    ActivationFunction *fn = new LinearActivation();
    int learningRate = 1.0f;

    int resultsSize = batchSize * dim.outputCubeSize;
    int inputSize = batchSize * dim.inputCubeSize;
    int weightsSize = dim.filtersSize;

    float *errors = new float[max(10000, resultsSize )];
    float *inputData = new float[max(10000, inputSize )];
    float *weights1 = new float[max(10000, weightsSize ) ];
    float *weights2 = new float[max(10000, weightsSize ) ];

    memset( errors, 0, sizeof(float) * max(10000, resultsSize ) );
    memset( inputData, 0, sizeof(float) * max(10000, inputSize ) );
    memset( weights1, 0, sizeof(float) * max(10000, weightsSize ) );
    memset( weights2, 0, sizeof(float) * max(10000, weightsSize ) );

//    WeightRandomizer::randomize( errors, max(10000, resultsSize ), 0.4, 1 );
//    WeightRandomizer::randomize( results, max( 10000, resultsSize), 1, 2 );
//    WeightRandomizer::randomize( inputData, max(10000, inputSize ), 0.2, 3 );

    WeightRandomizer::randomizeInts( errors, max(10000, resultsSize ), 1, 3 );
    WeightRandomizer::randomizeInts( inputData, max(10000, inputSize ), 1, 3 );

    OpenCLHelper cl;
    
    BackpropWeights2 *backpropWeightsImpl1 = BackpropWeights2::instanceSpecific( 0, &cl, dim );
    backpropWeightsImpl1->debug = true;
    backpropWeightsImpl1->backpropWeights( batchSize, learningRate,
        errors, inputData, weights1, 0 );
    BackpropWeights2 *backpropWeightsImpl2 = BackpropWeights2::instanceSpecific( 1, &cl, dim );
    backpropWeightsImpl2->debug = true;
    backpropWeightsImpl2->backpropWeights( batchSize, learningRate, 
        errors, inputData, weights2, 0 );

    cout << dim << endl;
    for( int i = 0; i < 25; i++ ) {
        cout << "weights[" << i << "]=" << weights1[i] << " " << weights2[i];
        if( i < weightsSize ) {
            if( abs( weights1[i] - weights2[i] ) <= abs(weights1[i]) / 10000.0f ) {
                cout << " SAME";
            } else {
                cout << " DIFF";
            }
        } else {
            cout << "     ";
        }
        cout << "  || " << weights2[100+i] ;
        cout << "  || " << weights2[200+i] ;
        cout << "  || " << weights2[300+i] ;
        cout << "  || " << weights2[400+i] ;
        cout << "  || " << weights2[500+i] ;
        cout << "  || " << weights2[600+i] ;
        cout << "  || " << weights2[700+i] << endl;
    }
    bool same = true;
    int errCount = 0;
    for( int i = 0; i < weightsSize; i++ ) {
        if( abs( weights1[i] - weights2[i] ) > abs(weights1[i]) / 10000.0f ) {
            cout << "DIFF: i " << i << " " << weights1[i] << " != " << weights2[i] << endl;
            same = false;
            errCount++;
            if( errCount == 5 ) {
                cout << " ... " << endl;
                break;
            }
        }
    }
    EXPECT_EQ( true, same );

    delete backpropWeightsImpl1;
    delete backpropWeightsImpl2;

    delete[] weights1;
    delete[] weights2;
    delete[] errors;
    delete[] inputData;
}

