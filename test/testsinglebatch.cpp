// tests the time for a single batch to go forward and backward through the layers

#include <iostream>
#include <random>

#include "NeuralNet.h"

#include "gtest/gtest.h"

#include "test/gtest_supp.h"
#include "test/Sampler.h"
#include "Timer.h"

using namespace std;

TEST( testsinglebatch, main ) {
    const int batchSize = 128;
    const float learningRate = 0.1f;

    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
    net->convolutionalMaker()->numFilters(10)->filterSize(20)->tanh()->biased()->insert();
    net->setBatchSize(batchSize);

    mt19937 random;
    random.seed(0); // so always gives same results
    const int inputsSize = net->getInputSizePerExample() * batchSize;
    float *inputData = new float[ inputsSize ];
    for( int i = 0; i < inputsSize; i++ ) {
        inputData[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
    }
    const int resultsSize = net->getLastLayer()->getResultsSize();
    float *expectedResults = new float[resultsSize];
    for( int i = 0; i < resultsSize; i++ ) {
        expectedResults[i] = random() / (float)mt19937::max() * 2.0f - 1.0f;
    }

    Timer timer;
    for( int i = 0; i < 5; i++ ) {
        net->learnBatch( learningRate, inputData, expectedResults );
    }
    timer.timeCheck("batch time");
    StatefulTimer::dump(true);

    Sampler::printSamples( "expectedResults", resultsSize, expectedResults );

    EXPECT_FLOAT_NEAR( 0.640024, expectedResults[684] );
    EXPECT_FLOAT_NEAR( -0.357197, expectedResults[559] );
    EXPECT_FLOAT_NEAR( 0.203795, expectedResults[373] );
    EXPECT_FLOAT_NEAR( -0.193317, expectedResults[960] );
    EXPECT_FLOAT_NEAR( -0.223511, expectedResults[323] );

    delete[] expectedResults;
    delete[] inputData;
    delete net;
}

