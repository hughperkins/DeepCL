// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <random>
#include <algorithm>

#include "NeuralNet.h"
#include "BackpropErrorsv2.h"
#include "ActivationFunction.h"
#include "LossLayer.h"
#include "ForceBackpropLayerMaker.h"

#include "gtest/gtest.h"

#include "test/gtest_supp.h"
#include "test/Sampler.h"
#include "test/WeightRandomizer.h"

using namespace std;

TEST( testbackward, squareloss ) {
    // here's the plan:
    // generate some input, randomly
    // generate some expected output, randomly
    // forward forward
    // calculate loss
    // calculate gradInput
    // change some of the inputs, forward prop, recalculate loss, check corresponds
    // to the gradient
    NeuralNet *net = new NeuralNet( 3, 5 );
    net->addLayer( ForceBackpropLayerMaker::instance() );
    net->addLayer( SquareLossMaker::instance() );
    cout << net->asString() << endl;

    int batchSize = 32;
    net->setBatchSize(batchSize);

    int inputCubeSize = net->getInputCubeSize();
    int outputCubeSize = net->getOutputCubeSize();

    int inputTotalSize = inputCubeSize * batchSize;
    int outputTotalSize = outputCubeSize * batchSize;

    cout << "inputtotalsize=" << inputTotalSize << " outputTotalSize=" << outputTotalSize << endl;

    float *input = new float[inputTotalSize];
    float *expectedOutput = new float[outputTotalSize];

    WeightRandomizer::randomize( 0, input, inputTotalSize, -2.0f, 2.0f );
    WeightRandomizer::randomize( 1, expectedOutput, outputTotalSize, -2.0f, 2.0f );
    
    // now, forward prop
//    net->input( input );
    net->forward( input );
    net->print();
//    net->printOutput();

    // calculate loss
    float lossBefore = net->calcLoss( expectedOutput );

    // calculate gradInput
    net->backward( 1.0f, expectedOutput);

    // modify input slightly
    mt19937 random;
    const int numSamples = 10;
    for( int i = 0; i < numSamples; i++ ) {
        int inputIndex;
        WeightRandomizer::randomizeInts( i, &inputIndex, 1, 0, inputTotalSize );
//        cout << "i=" << i << " index " << inputIndex << endl;
        float oldValue = input[inputIndex];
        // grad for this index is....
        float grad = net->getLayer(2)->getGradInput()[inputIndex];
//        cout << "grad=" << grad << endl;
        // tweak slightly
        float newValue = oldValue * 1.01f;
        float inputDelta = newValue - oldValue;
        float predictedLossChange = inputDelta * grad;
        input[inputIndex] = newValue;
//        cout << "oldvalue=" << oldValue << " newvalue=" << newValue << endl;
        // forwardProp
        net->forward( input );
        input[inputIndex] = oldValue;
//        net->printOutput();
        float lossAfter = net->calcLoss( expectedOutput );
        float lossChange = lossAfter - lossBefore;
        cout << "idx=" << inputIndex << " predicted losschange=" << predictedLossChange << " actual=" << lossChange << endl;
    }

    delete[] expectedOutput;
    delete[] input;

    delete net;
}

TEST( testbackward, crossentropyloss ) {
    // here's the plan:
    // generate some input, randomly
    // generate some expected output, randomly
    // forward forward
    // calculate loss
    // calculate gradInput
    // change some of the inputs, forward prop, recalculate loss, check corresponds
    // to the gradient
    NeuralNet *net = new NeuralNet( 3, 5 );
    net->addLayer( ForceBackpropLayerMaker::instance() );
    net->addLayer( CrossEntropyLossMaker::instance() );
    cout << net->asString() << endl;

    int batchSize = 32;
    net->setBatchSize(batchSize);

    int inputCubeSize = net->getInputCubeSize();
    int outputCubeSize = net->getOutputCubeSize();

    int inputTotalSize = inputCubeSize * batchSize;
    int outputTotalSize = outputCubeSize * batchSize;

    cout << "inputtotalsize=" << inputTotalSize << " outputTotalSize=" << outputTotalSize << endl;

    float *input = new float[inputTotalSize];
    float *expectedOutput = new float[outputTotalSize];

    WeightRandomizer::randomize( 0, input, inputTotalSize, -2.0f, 2.0f );
    WeightRandomizer::randomize( 1, expectedOutput, outputTotalSize, -2.0f, 2.0f );
    
    // now, forward prop
//    net->input( input );
    net->forward( input );
    net->print();
//    net->printOutput();

    // calculate loss
    float lossBefore = net->calcLoss( expectedOutput );

    // calculate gradInput
    net->backward( 1.0f, expectedOutput);

    // modify input slightly
    mt19937 random;
    const int numSamples = 10;
    for( int i = 0; i < numSamples; i++ ) {
        int inputIndex;
        WeightRandomizer::randomizeInts( i, &inputIndex, 1, 0, inputTotalSize );
//        cout << "i=" << i << " index " << inputIndex << endl;
        float oldValue = input[inputIndex];
        // grad for this index is....
        float grad = net->getLayer(2)->getGradInput()[inputIndex];
//        cout << "grad=" << grad << endl;
        // tweak slightly
        float newValue = oldValue * 1.01f;
        float inputDelta = newValue - oldValue;
        float predictedLossChange = inputDelta * grad;
        input[inputIndex] = newValue;
//        cout << "oldvalue=" << oldValue << " newvalue=" << newValue << endl;
        // forwardProp
        net->forward( input );
        input[inputIndex] = oldValue;
//        net->printOutput();
        float lossAfter = net->calcLoss( expectedOutput );
        float lossChange = lossAfter - lossBefore;
        cout << "idx=" << inputIndex << " predicted losschange=" << predictedLossChange << " actual=" << lossChange << endl;
    }

    delete[] expectedOutput;
    delete[] input;

    delete net;
}

// This file contains tests for calculating errors for the upstream layer

void testNumerically( float learningRate, int batchSize, int imageSize, int filterSize, int numPlanes, ActivationFunction *fn, bool padZeros, int its = 20 ) {
    NeuralNet *net = NeuralNet::maker()->planes(numPlanes)->imageSize(imageSize)->instance();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(1)->filterSize(filterSize)->biased(0)->padZeros(padZeros) );
    net->addLayer( ActivationMaker::instance()->fn(fn) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(1)->filterSize(filterSize)->biased(0)->padZeros(padZeros) );
    net->addLayer( ActivationMaker::instance()->fn(fn) );
    net->addLayer( SquareLossMaker::instance() );;
    net->setBatchSize( batchSize );

    int inputSize = net->getLayer(0)->getOutputSize();
    int outputSize = net->getLayer(3)->getOutputSize();
    int weightsSize1 = net->getLayer(1)->getWeightsSize();
    int weightsSize2 = net->getLayer(3)->getWeightsSize();

    float *inputData = new float[std::max<int>(10000, inputSize )];
    float *expectedOutput = new float[std::max<int>(10000, outputSize )];
    memset( inputData, 0, sizeof(float) * std::max<int>(10000, inputSize ) );
    memset( expectedOutput, 0, sizeof(float) * std::max<int>(10000, outputSize ) );
//    int seed = 0;
    std::mt19937 random = WeightRandomizer::randomize( inputData, std::max<int>(10000, inputSize ), -2.0f, 2.0f );
    WeightRandomizer::randomize( random, expectedOutput, std::max<int>(10000, outputSize ), -2.0f, 2.0f );
    WeightRandomizer::randomize( random, dynamic_cast<ConvolutionalLayer*>(net->getLayer(1))->weights, weightsSize1, -2.0f, 2.0f );
    dynamic_cast<ConvolutionalLayer*>(net->getLayer(1))->weightsWrapper->copyToDevice();
    WeightRandomizer::randomize( random, dynamic_cast<ConvolutionalLayer*>(net->getLayer(3))->weights, weightsSize2, -2.0f, 2.0f );
    dynamic_cast<ConvolutionalLayer*>(net->getLayer(3))->weightsWrapper->copyToDevice();

    for( int it = 0; it < its; it++ ) {
        float *weightsBefore1 = new float[weightsSize1];
        float *currentWeights = net->getLayer(1)->getWeights();
        for( int i = 0; i < weightsSize1; i++ ) {
            weightsBefore1[i] = currentWeights[i];
        }
        float *weightsBefore2 = new float[weightsSize2];
        currentWeights = net->getLayer(3)->getWeights();
        for( int i = 0; i < weightsSize2; i++ ) {
            weightsBefore2[i] = currentWeights[i];
        }

        net->forward( inputData );
    //    net->print();
        float loss = net->calcLoss(expectedOutput);
        dynamic_cast<LossLayer*>(net->getLayer(5))->calcLoss(expectedOutput);
        net->backward( learningRate, expectedOutput );
        dynamic_cast<ConvolutionalLayer*>(net->getLayer(1))->weightsWrapper->copyToHost();
        // restore 2nd layer weights :-)
        for( int i = 0; i < weightsSize2; i++ ) {
//            dynamic_cast<ConvolutionalLayer*>(net->getLayer(2))->weights[i] = weightsBefore2[i];
        }
        dynamic_cast<ConvolutionalLayer*>(net->getLayer(3))->weightsWrapper->copyToDevice();
        net->forward( inputData );

        float loss2 = net->calcLoss(expectedOutput);
        float lossChange = loss - loss2;
        cout << " loss " << loss << " loss2 " << loss2 << " change: " << lossChange << endl;

        float *newWeights = net->getLayer(1)->getWeights();
        float sumWeightDiff = 0;
        float sumWeightDiffSquared = 0;
        for( int i = 0; i < weightsSize1; i++ ) {
            float diff = newWeights[i] - weightsBefore1[i];
            sumWeightDiff += diff;
            sumWeightDiffSquared += diff * diff;
        }
        newWeights = net->getLayer(3)->getWeights();
        for( int i = 0; i < weightsSize2; i++ ) {
            float diff = newWeights[i] - weightsBefore2[i];
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
    }

//    delete[] weights1;
//    delete[] errors;
//    delete[] output;
    delete[] inputData;
}

TEST( testbackward, checknumerically ) {
    float learningRate = 0.1f;
    const int batchSize = 1;
    const int imageSize = 1;
    const int filterSize = 1;
    const int numPlanes = 1;
    bool padZeros = false;

    testNumerically( learningRate, batchSize, imageSize, filterSize, numPlanes, new TanhActivation(), padZeros, 5 );
}

TEST( testbackward, checknumerically_imagesize5_filter3_relu ) {
    float learningRate = 0.0001f;
    const int batchSize = 1;
    const int imageSize = 5;
    const int filterSize = 3;
    const int numPlanes = 1;
    ActivationFunction *fn = new ReluActivation();
    bool padZeros = true;

    testNumerically( learningRate, batchSize, imageSize, filterSize, numPlanes, fn, padZeros );
}

void measurePerf( int instance, int batchSize, LayerDimensions dim ) {
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    int inputSize = dim.inputCubeSize * batchSize;
    int errorsSize = dim.outputCubeSize * batchSize;
    int weightsSize = dim.filtersSize;
    int errorsForUpstreamSize = dim.inputCubeSize * batchSize;
    float *input = new float[inputSize];
    float *errors = new float[errorsSize];
    float *weights = new float[weightsSize];

    WeightRandomizer::randomize( input, inputSize, -0.1f, 0.1f );
    WeightRandomizer::randomize( errors, errorsSize, -0.1f, 0.1f );
    WeightRandomizer::randomize( weights, weightsSize, -0.1f, 0.1f );

    float *errorsForUpstream = new float[errorsForUpstreamSize];
    CLWrapper *inputWrapper = cl->wrap( inputSize, input );
    CLWrapper *errorsWrapper = cl->wrap( errorsSize, errors );
    CLWrapper *weightsWrapper = cl->wrap( weightsSize, weights );
    CLWrapper *errorsForUpstreamWrapper = cl->wrap( errorsForUpstreamSize, errorsForUpstream );
    inputWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    weightsWrapper->copyToDevice();
    errorsForUpstreamWrapper->createOnDevice();

    StatefulTimer::timeCheck("after init");
    BackpropErrorsv2 *backwardImpl = BackpropErrorsv2::instanceSpecific( instance, cl, dim );
    for( int it = 0; it < 40; it++ ) {
        backwardImpl->backward( batchSize, 
            inputWrapper, errorsWrapper, weightsWrapper,
            errorsForUpstreamWrapper );
    }
    StatefulTimer::timeCheck("after backprop");
    StatefulTimer::dump(true);

    delete errorsForUpstreamWrapper;
    delete weightsWrapper;
    delete inputWrapper;
    delete errorsWrapper;

    delete[] errors;
    delete[] weights;
    delete[] input;
    delete[] errorsForUpstream;

    delete backwardImpl;
    delete cl;
}

TEST( SLOW_testbackward, perf_kgsgo_32c5 ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputImageSize(19).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );  
    cout << dim.buildOptionsString() << endl;  
//    ActivationFunction *fn = new ReluActivation();

    measurePerf( 2, batchSize, dim );
}

void compareSpecific( int instance0, int instance1, int batchSize, LayerDimensions dim ) {
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    int inputSize = dim.inputCubeSize * batchSize;
    int errorsSize = dim.outputCubeSize * batchSize;
    int weightsSize = dim.filtersSize;
    int errorsForUpstreamSize = dim.inputCubeSize * batchSize;

    float *input = new float[inputSize];
    float *errors = new float[errorsSize];
    float *weights = new float[weightsSize];
    float *errorsForUpstream0 = new float[errorsForUpstreamSize];
    float *errorsForUpstream1 = new float[errorsForUpstreamSize];

    WeightRandomizer::randomize( input, inputSize, -0.1f, 0.1f );
    WeightRandomizer::randomize( errors, errorsSize, -0.1f, 0.1f );
    WeightRandomizer::randomize( weights, weightsSize, -0.1f, 0.1f );

    CLWrapper *inputWrapper = cl->wrap( inputSize, input );
    CLWrapper *errorsWrapper = cl->wrap( errorsSize, errors );
    CLWrapper *weightsWrapper = cl->wrap( weightsSize, weights );
    CLWrapper *errorsForUpstreamWrapper0 = cl->wrap( errorsForUpstreamSize, errorsForUpstream0 );
    CLWrapper *errorsForUpstreamWrapper1 = cl->wrap( errorsForUpstreamSize, errorsForUpstream1 );

    inputWrapper->copyToDevice();
    errorsWrapper->copyToDevice();
    weightsWrapper->copyToDevice();
    errorsForUpstreamWrapper0->createOnDevice();
    errorsForUpstreamWrapper1->createOnDevice();

    BackpropErrorsv2 *bp0 = BackpropErrorsv2::instanceSpecific( instance0, cl, dim );
    BackpropErrorsv2 *bp1 = BackpropErrorsv2::instanceSpecific( instance1, cl, dim );
    
    bp0->backward( batchSize, 
            inputWrapper, errorsWrapper, weightsWrapper,
            errorsForUpstreamWrapper0 );
    bp1->backward( batchSize, 
            inputWrapper, errorsWrapper, weightsWrapper,
            errorsForUpstreamWrapper1 );

    errorsForUpstreamWrapper0->copyToHost();
    errorsForUpstreamWrapper1->copyToHost();

    int outputSize = errorsForUpstreamSize;
    cout << dim << endl;
    bool same = true;
    for( int i = 0; i < max( 20, outputSize ); i++ ) {
        if( i < outputSize ) {
            if( abs( errorsForUpstream0[i] - errorsForUpstream1[i] ) < 0.000001 || abs( errorsForUpstream0[i] - errorsForUpstream1[i] ) <= 0.001 * max( abs( errorsForUpstream0[i] ), abs( errorsForUpstream1[i] ) ) ) {
                if( i < 20 ) {
                    cout << "output[" << i << "]=" << errorsForUpstream0[i] << " " << errorsForUpstream1[i];
                    cout << " SAME";
                }
            } else {
                cout << "output[" << i << "]=" << errorsForUpstream0[i] << " " << errorsForUpstream1[i];
                cout << " DIFF";
                same = false;
            }
        } else {
             if( i < 20 ) {
                 cout << "     ";
             }
        }
        if( i < 20 ) {
            cout << "  || " << errorsForUpstream1[100+i] ;
            cout << "  || " << errorsForUpstream1[200+i] ;
            cout << "  || " << errorsForUpstream1[300+i] ;
            cout << "  || " << errorsForUpstream1[400+i] ;
            cout << "  || " << errorsForUpstream1[500+i] ;
            cout << "  || " << errorsForUpstream1[600+i] ;
            cout << "  || " << errorsForUpstream1[700+i] << endl;
        }
    }
    EXPECT_EQ( true, same );

    delete inputWrapper;
    delete errorsWrapper;
    delete weightsWrapper;
    delete errorsForUpstreamWrapper0;
    delete errorsForUpstreamWrapper1;

    delete[] errorsForUpstream0;
    delete[] errorsForUpstream1;
    delete bp0;
    delete bp1;
    delete cl;
    delete[] input;
    delete[] errors;
    delete[] weights;
}

TEST( SLOW_testbackward, compare_kgsgo_32c5 ) {
    int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputImageSize(19).setNumFilters( 32 ).setFilterSize( 5 )
        .setPadZeros( true ).setBiased( true );  
    cout << dim.buildOptionsString() << endl;  
//    ActivationFunction *fn = new ReluActivation();

    compareSpecific( 1, 2, batchSize, dim );

}

TEST( SLOW_testbackward, compare_kgsgo_32c5mini ) {
    int batchSize = 4;
    LayerDimensions dim;
    dim.setInputPlanes( 2 ).setInputImageSize(3).setNumFilters( 2 ).setFilterSize( 3 )
        .setPadZeros( true ).setBiased( true );  
    cout << dim.buildOptionsString() << endl;  
//    ActivationFunction *fn = new ReluActivation();

    compareSpecific( 1, 2, batchSize, dim );

}

TEST( SLOW_testbackward, compare_kgsgo_32c5mini2 ) {
    int batchSize = 1;
    int imageSize = 2;
    LayerDimensions dim;
    dim.setInputPlanes( 1 ).setInputImageSize(imageSize).setNumFilters( 1 ).setFilterSize( imageSize )
        .setPadZeros( true ).setBiased( true );
    cout << dim.buildOptionsString() << endl;
//    ActivationFunction *fn = new ReluActivation();

    compareSpecific( 1, 2, batchSize, dim );

}

/*
float *test( int imageSize ) {
    const int batchSize = 128;
    LayerDimensions dim;
    dim.setInputPlanes( 32 ).setInputImageSize( 28 ).setNumFilters( 32 ).setFilterSize( 5 )
        .setBiased( true ).setPadZeros( true );

    int weightsSize = dim.filtersSize;
    int biasWeightsSize = dim.numFilters;
    int outputSize = batchSize * dim.outputCubeSize;
    float *weights = new float[max(10000, weightsSize ) ];
    float *biasWeights = new float[max( 10000, biasWeightsSize)];
    float *errors = new float[max(10000, outputSize )];
    float *output = new float[max(10000, outputSize )];
    WeightRandomizer::randomize( weights, max(10000, weightsSize ), -1, 1 );
    WeightRandomizer::randomize( biasWeights, max( 10000, biasWeightsSize), -1, 1 );
    WeightRandomizer::randomize( errors, max(10000, outputSize ), -1, 1 );
    WeightRandomizer::randomize( output, max(10000, outputSize ), -1, 1 );

    OpenCLHelper cl;
    BackpropErrorsv2 *backwardImpl = BackpropErrorsv2::instanceForTest( &cl, dim, new ReluActivation() );
    Timer timer;
    float *errorsForUpstream = backwardImpl->backward( batchSize, output, weights, biasWeights, errors );
    StatefulTimer::dump(true);
    timer.timeCheck("after calcing errors");

    Sampler::printSamples( "errorsForUpstream", batchSize * dim.inputCubeSize, errorsForUpstream );

    delete backwardImpl;

    delete[] errors;
    delete[] weights;
    delete[] biasWeights;

    return errorsForUpstream;
}
*/
// we want to test calcerrors for layer 2 in a network like:
//    NeuralNet *net = NeuralNet::maker()->planes(1)->imageSize(28)->instance();
//    net->addLayer( ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
//    net->addLayer( ConvolutionalMaker::instance()->numFilters(32)->filterSize(5)->relu()->biased()->insert();
//    net->addLayer( ConvolutionalMaker::instance()->numFilters(10)->filterSize(20)->tanh()->biased(config.biased)->insert();
//TEST( testbackward, DISABLED_image28 ) {
//    float *errorsForUpstream = test(28);
//    EXPECT_FLOAT_NEAR( -1.66007, errorsForUpstream[68268] );
//    EXPECT_FLOAT_NEAR( 0.823709, errorsForUpstream[2927151] );
//    EXPECT_FLOAT_NEAR( 6.99365, errorsForUpstream[1746549] );
//    EXPECT_FLOAT_NEAR( 7.25249, errorsForUpstream[576704] );
//    EXPECT_FLOAT_NEAR( 7.88787, errorsForUpstream[570179] );
//    delete[] errorsForUpstream;
//}

//TEST( testbackward, DISABLED_image19 ) { // make it work for a image19 first :-)
//    float *errorsForUpstream = test(19);
//    EXPECT_FLOAT_NEAR( -24.5602, errorsForUpstream[158380] );
//    EXPECT_FLOAT_NEAR( 7.39012, errorsForUpstream[2607] );
//    EXPECT_FLOAT_NEAR( -6.50315, errorsForUpstream[546421] );
//    EXPECT_FLOAT_NEAR( -1.22025, errorsForUpstream[429248] );
//    EXPECT_FLOAT_NEAR( -8.89935, errorsForUpstream[1200963] );
//    delete[] errorsForUpstream;

//    const int batchSize = 128;
//    LayerDimensions dim;
//    dim.setInputPlanes( 32 ).setInputImageSize( 19 ).setNumFilters( 32 ).setFilterSize( 5 )
//        .setBiased( true ).setPadZeros( true );    const int batchSize = 128;
//    LayerDimensions dim;
//    dim.setInputPlanes( 32 ).setInputImageSize( 28 ).setNumFilters( 32 ).setFilterSize( 5 )
//        .setBiased( true ).setPadZeros( true );

//    int weightsSize = dim.filtersSize;
//    int biasWeightsSize = dim.numFilters;
//    int outputSize = batchSize * dim.outputCubeSize;
//    float *weights = new float[max(10000, weightsSize ) ];
//    float *biasWeights = new float[max( 10000, biasWeightsSize)];
//    float *errors = new float[max(10000, outputSize )];
//    float *output = new float[max(10000, outputSize )];
//    WeightRandomizer::randomize( weights, max(10000, weightsSize ), -1, 1 );
//    WeightRandomizer::randomize( biasWeights, max( 10000, biasWeightsSize), -1, 1 );
//    WeightRandomizer::randomize( errors, max(10000, outputSize ), -1, 1 );
//    WeightRandomizer::randomize( output, max(10000, outputSize ), -1, 1 );

//    OpenCLHelper cl;
//    BackpropErrors *backwardImpl = BackpropErrors::instanceForTest( &cl, dim, new ReluActivation() );
//    Timer timer;
//    float *errorsForUpstream = backwardImpl->backward( batchSize, output, weights, biasWeights, errors );
//    StatefulTimer::dump(true);
//    timer.timeCheck("after calcing errors");

//    Sampler::printSamples( "errorsForUpstream", batchSize * dim.inputCubeSize, errorsForUpstream );

//    EXPECT_FLOAT_NEAR( -1.66007, errorsForUpstream[68268] );
//    EXPECT_FLOAT_NEAR( 0.823709, errorsForUpstream[2927151] );
//    EXPECT_FLOAT_NEAR( 6.99365, errorsForUpstream[1746549] );
//    EXPECT_FLOAT_NEAR( 7.25249, errorsForUpstream[576704] );
//    EXPECT_FLOAT_NEAR( 7.88787, errorsForUpstream[570179] );

//    delete backwardImpl;

//    delete[] errorsForUpstream;
//    delete[] errors;
//    delete[] weights;
//    delete[] biasWeights;


//    int weightsSize = dim.filtersSize;
//    int biasWeightsSize = dim.numFilters;
//    int outputSize = batchSize * dim.outputCubeSize;
//    float *weights = new float[max(10000, weightsSize ) ];
//    float *biasWeights = new float[max( 10000, biasWeightsSize)];
//    float *errors = new float[max(10000, outputSize )];
//    float *output = new float[max(10000, outputSize )];
//    WeightRandomizer::randomize( weights, max(10000, weightsSize ), -1, 1 );
//    WeightRandomizer::randomize( biasWeights, max( 10000, biasWeightsSize), -1, 1 );
//    WeightRandomizer::randomize( errors, max(10000, outputSize ), -1, 1 );
//    WeightRandomizer::randomize( output, max(10000, outputSize ), -1, 1 );

//    OpenCLHelper cl;
//    BackpropErrors *backwardImpl = BackpropErrors::instanceForTest( &cl, dim, new ReluActivation() );
//    Timer timer;
//    float *errorsForUpstream = backwardImpl->backward( batchSize, output, weights, biasWeights, errors );
//    StatefulTimer::dump(true);
//    timer.timeCheck("after calcing errors");

//    Sampler::printSamples( "errorsForUpstream", batchSize * dim.inputCubeSize, errorsForUpstream );

//    EXPECT_FLOAT_NEAR( -24.5602, errorsForUpstream[158380] );
//    EXPECT_FLOAT_NEAR( 7.39012, errorsForUpstream[2607] );
//    EXPECT_FLOAT_NEAR( -6.50315, errorsForUpstream[546421] );
//    EXPECT_FLOAT_NEAR( -1.22025, errorsForUpstream[429248] );
//    EXPECT_FLOAT_NEAR( -8.89935, errorsForUpstream[1200963] );

//    delete backwardImpl;

//    delete[] errorsForUpstream;
//    delete[] errors;
//    delete[] weights;
//    delete[] biasWeights;
//}

/*
TEST( testbackward, comparespecific ) {
    const int batchSize = 5;
    LayerDimensions dim;
    dim.setInputPlanes( 1 ).setInputImageSize( 5 ).setNumFilters( 1 ).setFilterSize( 3 )
        .setBiased( true ).setPadZeros( false );

    int weightsSize = dim.filtersSize;
    int biasWeightsSize = dim.numFilters;
    int outputSize = batchSize * dim.outputCubeSize;
    float *weights = new float[max(10000, weightsSize ) ];
    float *biasWeights = new float[max( 10000, biasWeightsSize)];
    float *errors = new float[max(10000, outputSize )];
    float *output = new float[max(10000, outputSize )];
    memset( weights, 0, sizeof(float) * max(10000, weightsSize ) );
    memset( biasWeights, 0, sizeof(float) * max(10000, biasWeightsSize ) );
    memset( errors, 0, sizeof(float) * max(10000, outputSize ) );
    memset( output, 0, sizeof(float) * max(10000, outputSize ) );
    mt19937 random = WeightRandomizer::randomize( weights, max(10000, weightsSize ), -1, 1 );
    WeightRandomizer::randomize( random, biasWeights, max( 10000, biasWeightsSize), -1, 1 );
    WeightRandomizer::randomize( random, errors, max(10000, outputSize ), -1, 1 );
    WeightRandomizer::randomize( random, output, max(10000, outputSize ), -1, 1 );
//    WeightRandomizer::randomizeInts( weights, max(10000, weightsSize ), 1, 3 );
//    WeightRandomizer::randomizeInts( biasWeights, max( 10000, biasWeightsSize), 0, 3 );
//    WeightRandomizer::randomizeInts( errors, max(10000, outputSize ), 0, 3 );
//    WeightRandomizer::randomizeInts( output, max(10000, outputSize ), 0, 3 );

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
    BackpropErrorsv2 *backwardImpl1 = BackpropErrorsv2::instanceSpecific( 0, &cl, dim, new ReluActivation() );
    float *errorsForUpstream1 = backwardImpl1->backward( batchSize, output, weights, biasWeights, errors );
    BackpropErrorsv2 *backwardImpl2 = BackpropErrorsv2::instanceSpecific( 1, &cl, dim, new ReluActivation() );
    float *errorsForUpstream2 = backwardImpl2->backward( batchSize, output, weights, biasWeights, errors );

    int errorsForUpstreamSize = batchSize * dim.inputCubeSize;
    cout << dim << endl;
    for( int i = 0; i < 25; i++ ) {
        cout << "output[" << i << "]=" << errorsForUpstream1[i] << " " << errorsForUpstream2[i];
        if( i < outputSize ) {
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

    delete backwardImpl1;
    delete backwardImpl2;

    delete[] errorsForUpstream1;
    delete[] errorsForUpstream2;
    delete[] errors;
    delete[] weights;
    delete[] biasWeights;
}
*/

