#include <iostream>
#include <random>

#include "NeuralNet.h"

#include "gtest/gtest.h"

#include "test/gtest_supp.h"
#include "test/Sampler.h"

using namespace std;

TEST( DISABLED_testbiasbackprop, one ) {
    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(28)->instance();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->convolutionalMaker()->numFilters(10)->filterSize(20)->tanh()->biased()->insert();
    net->setBatchSize(128);
    ConvolutionalLayer *layer = dynamic_cast< ConvolutionalLayer *>( net->layers[2] );
    const int outputSize = layer->getOutputSize();
    const int biasWeightsSize = layer->getBiasWeightsSize();

    mt19937 random;
    random.seed(0); // so always gives same output
    float *errors = new float[ outputSize ];
    float *output = new float[outputSize];
    for( int i = 0; i < outputSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
        output[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    float *biasWeightChanges = new float[biasWeightsSize];

    Timer timer;
    for( int i = 0; i < 200; i++ ) {
//        layer->doBiasBackpropCpu( 0.7f, output, errors, biasWeightChanges );
    }
    timer.timeCheck("after calcing bias change");

    Sampler::printSamples( "biasWeightChanges", biasWeightsSize, biasWeightChanges );

    EXPECT_FLOAT_NEAR( -0.00748203, biasWeightChanges[12] );
    EXPECT_FLOAT_NEAR( 0.0101042, biasWeightChanges[15] );
    EXPECT_FLOAT_NEAR( 0.0166035, biasWeightChanges[21] );
    EXPECT_FLOAT_NEAR( -0.0248314, biasWeightChanges[0] );
    EXPECT_FLOAT_NEAR( 0.0429298, biasWeightChanges[3] );

    delete[] biasWeightChanges;
    delete[] output;
    delete[] errors;
    delete net;    
}

