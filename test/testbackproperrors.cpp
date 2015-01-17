#include <iostream>
#include <random>

#include "NeuralNet.h"
#include "BackpropErrors.h"
#include "ActivationFunction.h"
#include "LossLayer.h"

#include "gtest/gtest.h"

#include "test/gtest_supp.h"
#include "test/Sampler.h"
#include "test/WeightRandomizer.h"

using namespace std;

// This file contains tests for calculating errors for the upstream layer

void testNumerically( float learningRate, int batchSize, int boardSize, int filterSize, int numPlanes, ActivationFunction *fn, bool padZeros ) {
    NeuralNet *net = NeuralNet::maker()->planes(numPlanes)->boardSize(boardSize)->instance();
    net->convolutionalMaker()->numFilters(1)->filterSize(filterSize)->biased(0)->fn(fn)->padZeros(padZeros)->insert();
    net->convolutionalMaker()->numFilters(1)->filterSize(filterSize)->biased(0)->fn(fn)->padZeros(padZeros)->insert();
    net->squareLossMaker()->insert();
    net->setBatchSize( batchSize );

    int inputSize = net->layers[0]->getResultsSize();
    int resultsSize = net->layers[2]->getResultsSize();
    int weightsSize1 = net->layers[1]->getWeightsSize();
    int weightsSize2 = net->layers[2]->getWeightsSize();

    float *inputData = new float[max(10000, inputSize )];
    float *expectedResults = new float[max(10000, resultsSize )];
    memset( inputData, 0, sizeof(float) * max(10000, inputSize ) );
    memset( expectedResults, 0, sizeof(float) * max(10000, resultsSize ) );
    int seed = 0;
    std::mt19937 random = WeightRandomizer::randomize( inputData, max(10000, inputSize ), -2.0f, 2.0f );
    WeightRandomizer::randomize( random, expectedResults, max(10000, resultsSize ), -2.0f, 2.0f );
    WeightRandomizer::randomize( random, dynamic_cast<ConvolutionalLayer*>(net->layers[1])->weights, weightsSize1, -2.0f, 2.0f );
    dynamic_cast<ConvolutionalLayer*>(net->layers[1])->weightsWrapper->copyToDevice();
    WeightRandomizer::randomize( random, dynamic_cast<ConvolutionalLayer*>(net->layers[2])->weights, weightsSize2, -2.0f, 2.0f );
    dynamic_cast<ConvolutionalLayer*>(net->layers[2])->weightsWrapper->copyToDevice();

    for( int it = 0; it < 20; it++ ) {
        float *weightsBefore1 = new float[weightsSize1];
        float *currentWeights = net->layers[1]->getWeights();
        for( int i = 0; i < weightsSize1; i++ ) {
            weightsBefore1[i] = currentWeights[i];
        }
        float *weightsBefore2 = new float[weightsSize2];
        currentWeights = net->layers[2]->getWeights();
        for( int i = 0; i < weightsSize2; i++ ) {
            weightsBefore2[i] = currentWeights[i];
        }

        net->propagate( inputData );
    //    net->print();
        float loss = net->calcLoss(expectedResults);
        float losslayer1 = dynamic_cast<LossLayer*>(net->layers[3])->calcLoss(expectedResults);
        net->backProp( learningRate, expectedResults );
        // restore 2nd layer weights :-)
        for( int i = 0; i < weightsSize2; i++ ) {
//            dynamic_cast<ConvolutionalLayer*>(net->layers[2])->weights[i] = weightsBefore2[i];
        }
        dynamic_cast<ConvolutionalLayer*>(net->layers[2])->weightsWrapper->copyToDevice();
        net->propagate( inputData );

        float loss2 = net->calcLoss(expectedResults);
        float lossChange = loss - loss2;
        cout << " loss " << loss << " loss2 " << loss2 << " change: " << lossChange << endl;

        float *newWeights = net->layers[1]->getWeights();
        float sumWeightDiff = 0;
        float sumWeightDiffSquared = 0;
        for( int i = 0; i < weightsSize1; i++ ) {
            float diff = newWeights[i] - weightsBefore1[i];
            sumWeightDiff += diff;
            sumWeightDiffSquared += diff * diff;
        }
        newWeights = net->layers[2]->getWeights();
        for( int i = 0; i < weightsSize2; i++ ) {
            float diff = newWeights[i] - weightsBefore2[i];
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
    }

//    delete[] weights1;
//    delete[] errors;
//    delete[] results;
    delete[] inputData;
}

TEST( testbackproperrors, checknumerically ) {
    float learningRate = 0.1f;
    const int batchSize = 1;
    const int boardSize = 1;
    const int filterSize = 1;
    const int numPlanes = 1;
    bool padZeros = false;

    testNumerically( learningRate, batchSize, boardSize, filterSize, numPlanes, new TanhActivation(), padZeros );
}

TEST( testbackproperrors, checknumerically_boardsize5_filter3_relu ) {
    float learningRate = 0.0001f;
    const int batchSize = 1;
    const int boardSize = 5;
    const int filterSize = 3;
    const int numPlanes = 1;
    ActivationFunction *fn = new ReluActivation();
    bool padZeros = true;

    testNumerically( learningRate, batchSize, boardSize, filterSize, numPlanes, fn, padZeros );
}

float *test( int boardSize ) {
    const int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputBoardSize( 28 ).setNumFilters( 32 ).setFilterSize( 5 )
        .setBiased( true ).setPadZeros( true );

    int weightsSize = dim.filtersSize;
    int biasWeightsSize = dim.numFilters;
    int resultsSize = batchSize * dim.outputCubeSize;
    float *weights = new float[max(10000, weightsSize ) ];
    float *biasWeights = new float[max( 10000, biasWeightsSize)];
    float *errors = new float[max(10000, resultsSize )];
    float *results = new float[max(10000, resultsSize )];
    WeightRandomizer::randomize( weights, max(10000, weightsSize ), -1, 1 );
    WeightRandomizer::randomize( biasWeights, max( 10000, biasWeightsSize), -1, 1 );
    WeightRandomizer::randomize( errors, max(10000, resultsSize ), -1, 1 );
    WeightRandomizer::randomize( results, max(10000, resultsSize ), -1, 1 );

    OpenCLHelper cl;
    BackpropErrors *backpropErrorsImpl = BackpropErrors::instanceForTest( &cl, dim, new ReluActivation() );
    Timer timer;
    float *errorsForUpstream = backpropErrorsImpl->backpropErrors( batchSize, results, weights, biasWeights, errors );
    StatefulTimer::dump(true);
    timer.timeCheck("after calcing errors");

    Sampler::printSamples( "errorsForUpstream", batchSize * dim.inputCubeSize, errorsForUpstream );

    delete backpropErrorsImpl;

    delete[] errors;
    delete[] weights;
    delete[] biasWeights;

    return errorsForUpstream;
}

// we want to test calcerrors for layer 2 in a network like:
//    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
//    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
//    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(20)->tanh()->biased(config.biased)->insert();
TEST( testbackproperrors, DISABLED_board28 ) {
    float *errorsForUpstream = test(28);
    EXPECT_FLOAT_NEAR( -1.66007, errorsForUpstream[68268] );
    EXPECT_FLOAT_NEAR( 0.823709, errorsForUpstream[2927151] );
    EXPECT_FLOAT_NEAR( 6.99365, errorsForUpstream[1746549] );
    EXPECT_FLOAT_NEAR( 7.25249, errorsForUpstream[576704] );
    EXPECT_FLOAT_NEAR( 7.88787, errorsForUpstream[570179] );
    delete[] errorsForUpstream;
}

TEST( testbackproperrors, DISABLED_board19 ) { // make it work for a board19 first :-)
    float *errorsForUpstream = test(19);
    EXPECT_FLOAT_NEAR( -24.5602, errorsForUpstream[158380] );
    EXPECT_FLOAT_NEAR( 7.39012, errorsForUpstream[2607] );
    EXPECT_FLOAT_NEAR( -6.50315, errorsForUpstream[546421] );
    EXPECT_FLOAT_NEAR( -1.22025, errorsForUpstream[429248] );
    EXPECT_FLOAT_NEAR( -8.89935, errorsForUpstream[1200963] );
    delete[] errorsForUpstream;

//    const int batchSize = 128;
//    LayerDimensions dim;
//    dim.setInputPlanes( 32 ).setInputBoardSize( 19 ).setNumFilters( 32 ).setFilterSize( 5 )
//        .setBiased( true ).setPadZeros( true );    const int batchSize = 128;
//    LayerDimensions dim;
//    dim.setInputPlanes( 32 ).setInputBoardSize( 28 ).setNumFilters( 32 ).setFilterSize( 5 )
//        .setBiased( true ).setPadZeros( true );

//    int weightsSize = dim.filtersSize;
//    int biasWeightsSize = dim.numFilters;
//    int resultsSize = batchSize * dim.outputCubeSize;
//    float *weights = new float[max(10000, weightsSize ) ];
//    float *biasWeights = new float[max( 10000, biasWeightsSize)];
//    float *errors = new float[max(10000, resultsSize )];
//    float *results = new float[max(10000, resultsSize )];
//    WeightRandomizer::randomize( weights, max(10000, weightsSize ), -1, 1 );
//    WeightRandomizer::randomize( biasWeights, max( 10000, biasWeightsSize), -1, 1 );
//    WeightRandomizer::randomize( errors, max(10000, resultsSize ), -1, 1 );
//    WeightRandomizer::randomize( results, max(10000, resultsSize ), -1, 1 );

//    OpenCLHelper cl;
//    BackpropErrors *backpropErrorsImpl = BackpropErrors::instanceForTest( &cl, dim, new ReluActivation() );
//    Timer timer;
//    float *errorsForUpstream = backpropErrorsImpl->backpropErrors( batchSize, results, weights, biasWeights, errors );
//    StatefulTimer::dump(true);
//    timer.timeCheck("after calcing errors");

//    Sampler::printSamples( "errorsForUpstream", batchSize * dim.inputCubeSize, errorsForUpstream );

//    EXPECT_FLOAT_NEAR( -1.66007, errorsForUpstream[68268] );
//    EXPECT_FLOAT_NEAR( 0.823709, errorsForUpstream[2927151] );
//    EXPECT_FLOAT_NEAR( 6.99365, errorsForUpstream[1746549] );
//    EXPECT_FLOAT_NEAR( 7.25249, errorsForUpstream[576704] );
//    EXPECT_FLOAT_NEAR( 7.88787, errorsForUpstream[570179] );

//    delete backpropErrorsImpl;

//    delete[] errorsForUpstream;
//    delete[] errors;
//    delete[] weights;
//    delete[] biasWeights;


//    int weightsSize = dim.filtersSize;
//    int biasWeightsSize = dim.numFilters;
//    int resultsSize = batchSize * dim.outputCubeSize;
//    float *weights = new float[max(10000, weightsSize ) ];
//    float *biasWeights = new float[max( 10000, biasWeightsSize)];
//    float *errors = new float[max(10000, resultsSize )];
//    float *results = new float[max(10000, resultsSize )];
//    WeightRandomizer::randomize( weights, max(10000, weightsSize ), -1, 1 );
//    WeightRandomizer::randomize( biasWeights, max( 10000, biasWeightsSize), -1, 1 );
//    WeightRandomizer::randomize( errors, max(10000, resultsSize ), -1, 1 );
//    WeightRandomizer::randomize( results, max(10000, resultsSize ), -1, 1 );

//    OpenCLHelper cl;
//    BackpropErrors *backpropErrorsImpl = BackpropErrors::instanceForTest( &cl, dim, new ReluActivation() );
//    Timer timer;
//    float *errorsForUpstream = backpropErrorsImpl->backpropErrors( batchSize, results, weights, biasWeights, errors );
//    StatefulTimer::dump(true);
//    timer.timeCheck("after calcing errors");

//    Sampler::printSamples( "errorsForUpstream", batchSize * dim.inputCubeSize, errorsForUpstream );

//    EXPECT_FLOAT_NEAR( -24.5602, errorsForUpstream[158380] );
//    EXPECT_FLOAT_NEAR( 7.39012, errorsForUpstream[2607] );
//    EXPECT_FLOAT_NEAR( -6.50315, errorsForUpstream[546421] );
//    EXPECT_FLOAT_NEAR( -1.22025, errorsForUpstream[429248] );
//    EXPECT_FLOAT_NEAR( -8.89935, errorsForUpstream[1200963] );

//    delete backpropErrorsImpl;

//    delete[] errorsForUpstream;
//    delete[] errors;
//    delete[] weights;
//    delete[] biasWeights;
}

TEST( testbackproperrors, comparespecific ) {
    const int batchSize = 5;
    LayerDimensions dim;
    dim.setInputPlanes( 1 ).setInputBoardSize( 5 ).setNumFilters( 1 ).setFilterSize( 3 )
        .setBiased( true ).setPadZeros( false );

    int weightsSize = dim.filtersSize;
    int biasWeightsSize = dim.numFilters;
    int resultsSize = batchSize * dim.outputCubeSize;
    float *weights = new float[max(10000, weightsSize ) ];
    float *biasWeights = new float[max( 10000, biasWeightsSize)];
    float *errors = new float[max(10000, resultsSize )];
    float *results = new float[max(10000, resultsSize )];
    memset( weights, 0, sizeof(float) * max(10000, weightsSize ) );
    memset( biasWeights, 0, sizeof(float) * max(10000, biasWeightsSize ) );
    memset( errors, 0, sizeof(float) * max(10000, resultsSize ) );
    memset( results, 0, sizeof(float) * max(10000, resultsSize ) );
    mt19937 random = WeightRandomizer::randomize( weights, max(10000, weightsSize ), -1, 1 );
    WeightRandomizer::randomize( random, biasWeights, max( 10000, biasWeightsSize), -1, 1 );
    WeightRandomizer::randomize( random, errors, max(10000, resultsSize ), -1, 1 );
    WeightRandomizer::randomize( random, results, max(10000, resultsSize ), -1, 1 );
//    WeightRandomizer::randomizeInts( weights, max(10000, weightsSize ), 1, 3 );
//    WeightRandomizer::randomizeInts( biasWeights, max( 10000, biasWeightsSize), 0, 3 );
//    WeightRandomizer::randomizeInts( errors, max(10000, resultsSize ), 0, 3 );
//    WeightRandomizer::randomizeInts( results, max(10000, resultsSize ), 0, 3 );

//    weights[0] = 3;
//    weights[1] = 5;
//    weights[2] = 4;

//    weights[25] = 4;
//    weights[49] = 4;

//    weights[50] = 4;
//    weights[99] = 4;

//    weights[75] = 4;
//    weights[99] = 4;

//    weights[100] = 3;
//    weights[124] = 3;

//    errors[0] = 2;
//    errors[1] = 7;
//    errors[2] = 3;
//    errors[3] = 1;
//    errors[4] = 8;
//    errors[5] = 6;

    OpenCLHelper cl;
    BackpropErrors *backpropErrorsImpl1 = BackpropErrors::instanceSpecific( 0, &cl, dim, new ReluActivation() );
    float *errorsForUpstream1 = backpropErrorsImpl1->backpropErrors( batchSize, results, weights, biasWeights, errors );
    BackpropErrors *backpropErrorsImpl2 = BackpropErrors::instanceSpecific( 1, &cl, dim, new ReluActivation() );
    float *errorsForUpstream2 = backpropErrorsImpl2->backpropErrors( batchSize, results, weights, biasWeights, errors );

    int errorsForUpstreamSize = batchSize * dim.inputCubeSize;
    cout << dim << endl;
    for( int i = 0; i < 25; i++ ) {
        cout << "results[" << i << "]=" << errorsForUpstream1[i] << " " << errorsForUpstream2[i];
        if( i < resultsSize ) {
            if( errorsForUpstream1[i] == errorsForUpstream2[i] ) {
                cout << " SAME";
            } else {
                cout << " DIFF";
            }
        } else {
            cout << "     ";
        }
        cout << "  || " << errorsForUpstream2[100+i] ;
        cout << "  || " << errorsForUpstream2[200+i] ;
        cout << "  || " << errorsForUpstream2[300+i] ;
        cout << "  || " << errorsForUpstream2[400+i] ;
        cout << "  || " << errorsForUpstream2[500+i] ;
        cout << "  || " << errorsForUpstream2[600+i] ;
        cout << "  || " << errorsForUpstream2[700+i] << endl;
    }
    bool same = true;
    int errCount = 0;
    for( int i = 0; i < errorsForUpstreamSize; i++ ) {
        if( errorsForUpstream1[i] != errorsForUpstream2[i] ) {
            cout << "DIFF: i " << i << " " << errorsForUpstream1[i] << " != " << errorsForUpstream2[i] << endl;
            same = false;
            errCount++;
            if( errCount == 5 ) {
                cout << " ... " << endl;
                break;
            }
        }
    }
    EXPECT_EQ( true, same );

    delete backpropErrorsImpl1;
    delete backpropErrorsImpl2;

    delete[] errorsForUpstream1;
    delete[] errorsForUpstream2;
    delete[] errors;
    delete[] weights;
    delete[] biasWeights;
}


