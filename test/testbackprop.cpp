// this will check the backprop process, and regression test it against original results, with
// original slow procedure

#include <iostream>
#include <random>

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

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

    int upstreamResultsSize = net->layers[0]->getResultsSize();
    float *upstreamResults = new float[upstreamResultsSize];
    for( int i = 0; i < upstreamResultsSize; i++ ) {
        upstreamResults[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    dynamic_cast<InputLayer*>(net->layers[0])->in(upstreamResults);

    int weightsSize = layer1->getWeightsSize();
    float *weights = new float[weightsSize];
    for( int i = 0; i < weightsSize; i++ ) {
        weights[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }
    net->initWeights( 1, weights );

    float *weightChanges = new float[weightsSize];

    int resultsSize = layer1->getResultsSize();
    float *errors = new float[resultsSize];
    for( int i = 0; i < resultsSize; i++ ) {
        errors[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
        layer1->results[i] = random() / (float)mt19937::max() * 0.2f - 0.1f;
    }

//    cout << " upstreamresultssize " << upstreamResultsSize << " resultsSize " << resultsSize <<
//         " weightsSize " << weightsSize << endl;
    for( int i = 0 ; i < 20; i++ ) {
        StatefulTimer::timeCheck("before backprop");
        layer1->backPropWeightsGpu( 0.1f, errors, weightChanges );
//        layer1->backPropWeightsCpu( 0.1f, errors, weightChanges );
//        cout << "after backprop" << endl;
        StatefulTimer::timeCheck("after backprop");
        random.seed(0);
//        for( int sample = 0; sample < 10; sample++ ) {
//            int index = random() % weightsSize;
//            cout << "weightChanges[" << index << "]=" << weightChanges[index] << endl;
//        }
//        EXPECT_FLOAT_NEAR(3.81361e-05,  weightChanges[33], 0.01f );
//        EXPECT_FLOAT_NEAR( -2.09972e-05,  weightChanges[144], 0.01f );
//        EXPECT_FLOAT_NEAR( 1.44103e-05,  weightChanges[339], 0.01f );
        EXPECT_NEAR(3.81361e-05,  weightChanges[33], 0.0000001f );
        EXPECT_NEAR( -2.09972e-05,  weightChanges[144], 0.0000001f );
        EXPECT_NEAR( 1.44103e-05,  weightChanges[339], 0.00000001f );
    }
    StatefulTimer::dump(true);
//    cout << "end" << endl;
    delete[] errors;
    delete[]weightChanges;
    delete[]weights;
    delete[] upstreamResults;
}
    
