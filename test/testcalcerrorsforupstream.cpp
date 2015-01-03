#include <iostream>
#include <random>

#include "NeuralNet.h"

#include "gtest/gtest.h"

#include "test/gtest_supp.h"
#include "test/Sampler.h"

using namespace std;

// This file contains tests for calculating errors for the upstream layer, whcih is currently a bit slow 
// as at the time of first creating this file :-P

// we want to test calcerrors for layer 2 in a network like:
//    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
//    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
//    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(20)->tanh()->biased(config.biased)->insert();
TEST( testcalcerrorsforupstream, one ) {
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->convolutionalMaker()->numFilters(10)->filterSize(20)->tanh()->biased()->insert();
    net->setBatchSize(128);

    ConvolutionalLayer *layer = dynamic_cast< ConvolutionalLayer *>( net->layers[2] );
    const int filterSize = layer->filterSize;
    const int filterSizeSquared = filterSize * filterSize;
    const int numOutPlanes = layer->numPlanes;
    const int upstreamNumPlanes = layer->upstreamNumPlanes;
    const int boardSize = layer->boardSize;
    const int upstreamBoardSize = layer->upstreamBoardSize;
    const int upstreamBoardSizeSquared = layer->upstreamBoardSizeSquared;
    const int upstreamResultsSize = layer->previousLayer->getResultsSize();
    const int resultsSize = layer->getResultsSize();
    const int weightsSize = layer->getWeightsSize();

    // need to set up:
    // weights (from our layer)
    // errors (from downstream)

    mt19937 random;
    random.seed(0); // so always gives same results
    float *errors = new float[ upstreamResultsSize ];
    for( int i = 0; i < resultsSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
    }
    for( int i = 0; i < weightsSize; i++ ) {
        layer->weights[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
    }
    float *errorsForUpstream = new float[upstreamResultsSize];

    Timer timer;
    layer->calcErrorsForUpstreamGpu( errors, errorsForUpstream );
    timer.timeCheck("after calcing errors");

    Sampler::printSamples( "errorsForUpstream", upstreamResultsSize, errorsForUpstream );

    EXPECT_FLOAT_NEAR( -36.2644, errorsForUpstream[199340] );
    EXPECT_FLOAT_NEAR( -6.55394, errorsForUpstream[567855] );
    EXPECT_FLOAT_NEAR( 33.4468, errorsForUpstream[2270837] );
    EXPECT_FLOAT_NEAR( 3.66922, errorsForUpstream[2215104] );
    EXPECT_FLOAT_NEAR( -17.9404, errorsForUpstream[701251] );

    delete[] errorsForUpstream;
    delete[] errors;
    delete net;    
}


