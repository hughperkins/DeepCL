#include <iostream>
#include <random>

#include "NeuralNet.h"

#include "gtest/gtest.h"

#include "test/gtest_supp.h"
#include "test/Sampler.h"

using namespace std;

TEST( testbiasbackprop, one ) {
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->convolutionalMaker()->numFilters(10)->filterSize(20)->tanh()->biased()->insert();
    net->setBatchSize(128);
    ConvolutionalLayer *layer = dynamic_cast< ConvolutionalLayer *>( net->layers[2] );
    const int resultsSize = layer->getResultsSize();
    const int biasWeightsSize = layer->getBiasWeightsSize();

    mt19937 random;
    random.seed(0); // so always gives same results
    float *errors = new float[ resultsSize ];
    float *results = new float[resultsSize];
    for( int i = 0; i < resultsSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
        results[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    float *biasWeightChanges = new float[biasWeightsSize];

    Timer timer;
    for( int i = 0; i < 200; i++ ) {
        layer->doBiasBackpropCpu( 0.7f, results, errors, biasWeightChanges );
    }
    timer.timeCheck("after calcing bias change");

    Sampler::printSamples( "biasWeightChanges", biasWeightsSize, biasWeightChanges );

    EXPECT_FLOAT_NEAR( -0.00748203, biasWeightChanges[12] );
    EXPECT_FLOAT_NEAR( 0.0101042, biasWeightChanges[15] );
    EXPECT_FLOAT_NEAR( 0.0166035, biasWeightChanges[21] );
    EXPECT_FLOAT_NEAR( -0.0248314, biasWeightChanges[0] );
    EXPECT_FLOAT_NEAR( 0.0429298, biasWeightChanges[3] );

    delete[] biasWeightChanges;
    delete[] results;
    delete[] errors;
    delete net;    
}

