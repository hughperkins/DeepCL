// this will check the backprop process, and regression test it against original results, with
// original slow procedure

#include <iostream>
#include <random>

#include "gtest/gtest.h"

#include "ConvolutionalLayer.h"
#include "NeuralNet.h"

using namespace std;

// this will test layer 1 backprop in a network like:
//    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
//    net->convolutionalMaker()->numFilters(14)->filterSize(5)->tanh()->biased()->insert();
//    net->convolutionalMaker()->numFilters(10)->filterSize(24)->tanh()->biased(config.biased)->insert();
TEST( testbackprop, main ) {
    mt19937 random;
    random.seed(0); // so always gives same results
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(28)->instance();
    net->convolutionalMaker()->numFilters(14)->filterSize(5)->tanh()->biased()->insert();
    net->convolutionalMaker()->numFilters(10)->filterSize(24)->tanh()->biased()->insert();
    net->setBatchSize(128);
    StatefulTimer::timeCheck("start");
    ConvolutionalLayer *layer1 = dynamic_cast<ConvolutionalLayer *>( net->layers[1] ); 
    int inputSize = net->layers[0]->getResultsSize();
    float *input = new float[inputSize];
    for( int i = 0; i < inputSize; i++ ) {
        input[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(input);
//    net->propagate( input );
    StatefulTimer::timeCheck("after forward-propagate");
    int resultsSize = layer1->getResultsSize();
    int weightsSize = layer1->getWeightsSize();
    float *errors = new float[resultsSize];
    float *weightChanges = new float[weightsSize];
    for( int i = 0; i < resultsSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    for( int i = 0 ; i < 20; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backPropWeightsGpu( 0.1f, errors, weightChanges );
        StatefulTimer::timeCheck("after backprop");
        random.seed(0);
//        for( int sample = 0; sample < 10; sample++ ) {
//            int index = random() % weightsSize;
//            cout << "weightChanges[" << index << "]=" << weightChanges[index] << endl;
//        }
        ASSERT_NEAR(-2.78677e-05,  weightChanges[33], 2.7e-5f * 0.001 );
        ASSERT_NEAR( -2.49555e-05,  weightChanges[144], 2.5e-5f * 0.001 );
        ASSERT_NEAR( -4.7395e-06,  weightChanges[339], 5e-6f * 0.001 );
    }
    StatefulTimer::dump(true);
    cout << "end" << endl;
}
    
