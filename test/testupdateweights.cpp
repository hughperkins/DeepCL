// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <iomanip>
#include <algorithm>

#include "EasyCL.h"
#include "net/NeuralNet.h"
#include "conv/BackpropWeights.h"
#include "conv/BackpropWeightsNaive.h"
#include "layer/Layer.h"
#include "conv/ConvolutionalLayer.h"
#include "conv/ConvolutionalMaker.h"
#include "net/NeuralNetMould.h"
#include "input/InputLayer.h"
#include "layer/LayerMakers.h"
#include "trainers/SGD.h"
#include "clblas/ClBlasInstance.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"
#include "test/DimFromArgs.h"
#include "test/TestArgsParser.h"

using namespace std;

void checkWeightsUpdate(NeuralNet *net, int targetLayerIndex) {
    // here's the plan:
    // generate some input, randomly
    // generate some expected output, randomly
    // forward propagate
    // calculate loss
    // calculate gradWeights
    // change some of the weights, forward prop, recalculate loss, check corresponds
    // to the gradient
    cout << net->asString() << endl;

    int batchSize = dynamic_cast< InputLayer *>(net->getLayer(0))->batchSize;
    cout << "batchSize: " << batchSize << endl;
//    const int outputPlanes = net->getOutputPlanes();

    int inputCubeSize = net->getInputCubeSize();
    int outputCubeSize = net->getOutputCubeSize();

    int inputTotalSize = inputCubeSize * batchSize;
    int outputTotalSize = outputCubeSize * batchSize;

    cout << "inputtotalsize=" << inputTotalSize << " outputTotalSize=" << outputTotalSize << endl;

    float *input = new float[inputTotalSize];
    float *expectedOutput = new float[outputTotalSize];
    Layer *layer = net->getLayer(targetLayerIndex);

    cout << "layer " << layer->asString() << endl;
    WeightRandomizer::randomize(1, input, inputTotalSize, -1.0f, 1.0f);
    WeightRandomizer::randomize(2, expectedOutput, outputTotalSize, -1.0f, 1.0f);

    int weightsSize = layer->getWeightsSize();
    int biasSize = layer->getBiasSize();
    cout << "weightsize=" << weightsSize << " biassize=" << biasSize << endl;
    float *weights = new float[weightsSize];
    WeightRandomizer::randomize(3, weights, weightsSize, -0.1f, 0.1f);
    float *bias = 0;
    if(layer->biased()) {
        bias = new float[biasSize];
        WeightRandomizer::randomize(4, bias, biasSize, -0.1f, 0.1f);
    }
    if(weightsSize > 0 || biasSize > 0) {
        layer->setWeights(weights, bias);
    }

    // now, forward prop
    net->forward(input);
    net->print();
//    net->printOutput();

    // calculate loss
    float lossBefore = net->calcLoss(expectedOutput);

    // calculate gradInput
    // should be zero, so we dont modify the weights
    // otherwise the losses will be really strange :-)
    // temporarily putting 1.0f, because of the way this works currently...
    net->backward(expectedOutput);

    // modify input slightly
    mt19937 random;
    const int numSamples = 10;
    for(int i = 0; i < numSamples; i++) {
        int weightIndex;
        WeightRandomizer::randomizeInts(i + 1, &weightIndex, 1, 0, weightsSize);
//        cout << "i=" << i << " index " << inputIndex << endl;
        float oldValue = weights[weightIndex];
        // grad for this index is....
        float grad = layer->getGradWeights()[weightIndex];
//        cout << "grad=" << grad << endl;
        // tweak slightly
        float newValue = oldValue * 1.01f;
        float inputDelta = newValue - oldValue;
        float predictedLossChange = inputDelta * grad;
        weights[weightIndex] = newValue;
        layer->setWeights(weights, bias);
//        cout << "oldvalue=" << oldValue << " newvalue=" << newValue << endl;
        // forwardProp
        net->forward(input);
        weights[weightIndex] = oldValue;
        layer->setWeights(weights, bias);
//        net->printOutput();
        float lossAfter = net->calcLoss(expectedOutput);
        float lossChange = lossAfter - lossBefore;
        cout << "idx=" << weightIndex << " predicted losschange=" << predictedLossChange << " actual=" << lossChange << endl;
    }

    delete[] weights;
    delete[] bias;
    delete[] expectedOutput;
    delete[] input;
}

TEST(testupdateweights, conv1) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = new NeuralNet(cl, 2, 5);
    net->addLayer(ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased(0)->padZeros(0));
    net->addLayer(SquareLossMaker::instance());
    cout << net->asString() << endl;

    net->setBatchSize(4);

    checkWeightsUpdate(net, 1);
    delete net;
    delete cl;
}

TEST(testupdateweights, conv1z) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = new NeuralNet(cl, 2, 3);
    net->addLayer(ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased(0)->padZeros(1));
    net->addLayer(SquareLossMaker::instance());
    cout << net->asString() << endl;

    net->setBatchSize(4);

    checkWeightsUpdate(net, 1);
    delete net;
    delete cl;
}

void test(int imageSize, int filterSize, int numPlanes, int batchSize) {
    float learningRate = 0.01f;

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
    net->addLayer(ConvolutionalMaker::instance()->numFilters(1)->filterSize(filterSize)->biased(0));
    net->addLayer(ActivationMaker::instance()->tanh());
    net->addLayer(SquareLossMaker::instance());;
    net->setBatchSize(batchSize);

    int inputNumElements = net->getLayer(0)->getOutputNumElements();
    int outputNumElements = net->getLayer(1)->getOutputNumElements();
    int weightsSize = net->getLayer(1)->getWeightsSize();

    float *inputData = new float[max(10000, inputNumElements)];
    float *expectedOutput = new float[max(10000, outputNumElements)];
    memset(inputData, 0, sizeof(float) * max(10000, inputNumElements));
    memset(expectedOutput, 0, sizeof(float) * max(10000, outputNumElements));
    std::mt19937 random = WeightRandomizer::randomize(inputData, max(10000, inputNumElements), -1.0f, 1.0f);
    WeightRandomizer::randomize(random, expectedOutput, max(10000, outputNumElements), -1.0f, 1.0f);
    WeightRandomizer::randomize(random, net->getLayer(1)->getWeights(), weightsSize, -0.1f, 0.1f);
    dynamic_cast<ConvolutionalLayer*>(net->getLayer(1))->weightsWrapper->copyToDevice();

    float *weightsBefore = new float[weightsSize];
    float const*currentWeights = net->getLayer(1)->getWeights();
    for(int i = 0; i < weightsSize; i++) {
        weightsBefore[i] = currentWeights[i];
    }

    net->forward(inputData);
    float loss = net->calcLoss(expectedOutput);

    net->print();
    SGD *sgd = SGD::instance(cl, learningRate, 0.0f);
    TrainingContext context(0, 0);
    sgd->train(net, &context, inputData, expectedOutput);
    net->forward(inputData);
    net->print();
    float loss2 = net->calcLoss(expectedOutput);
    float lossChange = loss - loss2;
    cout << " loss " << loss << " loss2 " << loss2 << " change: " << lossChange << endl;

    dynamic_cast<ConvolutionalLayer*>(net->getLayer(1))->weightsWrapper->copyToHost();
    float const*newWeights = net->getLayer(1)->getWeights();
    float sumWeightDiff = 0;
    float sumWeightDiffSquared = 0;
    for(int i = 0; i < weightsSize; i++) {
        float diff = newWeights[i] - weightsBefore[i];
        sumWeightDiff += diff;
        sumWeightDiffSquared += diff * diff;
    }
    cout << "sumweightsdiff " << sumWeightDiff << endl;

    float estimatedLossChangeFromW = sumWeightDiffSquared/ learningRate; // / filterSize;

    cout << " loss change              " << lossChange << endl;
    cout << " estimatedLossChangeFromW " << estimatedLossChangeFromW << endl;
    EXPECT_GT(0.01f * imageSize * imageSize, abs(estimatedLossChangeFromW - lossChange) / lossChange); 
    EXPECT_GT(0.01f * imageSize * imageSize, abs(estimatedLossChangeFromW - lossChange) / estimatedLossChangeFromW); 

    delete[] weightsBefore;
    delete sgd;
    delete net;
    delete[] expectedOutput;
    delete[] inputData;
    delete cl;
}

TEST(testupdateweights, numericallytest) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(1, 1, 1, 1);
}

TEST(testupdateweights, numericallytest_imagesize3) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(3, 1, 1, 1);
}

TEST(testupdateweights, numericallytest_imagesize5) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 1, 1, 1);
}

TEST(testupdateweights, numericallytest_imagesize9) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 1, 1, 1);
}

TEST(testupdateweights, numericallytest_imagesize9_filtersize9) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 9, 1, 1);
}

TEST(testupdateweights, numericallytest_imagesize9_filtersize3) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(9, 3, 1, 1);
}

TEST(testupdateweights, numericallytest_imagesize3_filtersize3) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(3, 3, 1, 1);
}

TEST(testupdateweights, numericallytest_imagesize5_filtersize3) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 1, 1);
}

TEST(testupdateweights, numericallytest_imagesize5_filtersize3_batchsize3) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 1, 3);
}

TEST(testupdateweights, numericallytest_imagesize5_filtersize3_planes3) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 3, 1);
}

TEST(testupdateweights, numericallytest_imagesize5_filtersize3_planes3_batchsize3) {
    // do one learning, with very small learning rate, and check that loss function changed by
    // the amount that we kind of expect
    test(5, 3, 3, 3);
}

void testBackpropWeights(LayerDimensions &dim, int batchSize, float learningMultiplier, float *data, float *errors, float * expectedOutput) {
    float *output = new float[batchSize * dim.outputCubeSize]; // ignored, for LINEAR
    float *weights = new float[max(dim.filtersSize,20)];
    float *bias = new float[10];
    memset(weights, 0, sizeof(float) * max(dim.filtersSize, 20));
    memset(bias, 0, sizeof(float) * 10);

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    BackpropWeights *backpropWeightsImpl = BackpropWeights::instanceForTest(cl, dim);
    backpropWeightsImpl->calcGradWeights(batchSize, errors, data, weights, bias);
    delete backpropWeightsImpl;
    
//    for(int i = 0; i < 20; i++) {
//        cout << "weights[" << i << "]=" << weights[i] << endl;
//    }
    for(int i = 0; i < dim.filtersSize; i++) {
        if(expectedOutput[i] != -999 && expectedOutput[i] != weights[i]) {
            cout << "mismatch for i " << i << endl;
            EXPECT_EQ(- expectedOutput[i], weights[i]);
        }
    }
    delete[] output;
    delete[] weights;
    delete[] bias;
    delete cl;
}

TEST(testupdateweights, backprop_weights_2) {
    LayerDimensions dim;
    dim.setInputSize(1).setInputPlanes(1).setNumFilters(1).setFilterSize(1)
        .setBiased(0).setPadZeros(0);

    const int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f };
    float errors[] = { 7.0f };
    float expectedOutput[] = { - 3 * 7 };
    testBackpropWeights(dim, batchSize, learningMultiplier, data, errors, expectedOutput);
}


TEST(testupdateweights, backprop_weights_2_upstreamimagesize2) {
    LayerDimensions dim;
    dim.setInputSize(2).setInputPlanes(1).setNumFilters(1).setFilterSize(1)
        .setBiased(0).setPadZeros(0);
    int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f, 13,
                    17, 19 };
    float DerivLossBySum[] = { 7.0f, 2,
                       4,4 };
    float expectedOutput[] = { -3 * 7 - 13 * 2 // -191
                                 -17*4 -19*4 };   // 

    testBackpropWeights(dim, batchSize, learningMultiplier, data, DerivLossBySum, expectedOutput);
}

TEST(testupdateweights, backprop_weights_2_upstreamimagesize3_filtersize3) {
    LayerDimensions dim;
    dim.setInputSize(3).setInputPlanes(1).setNumFilters(1).setFilterSize(3)
        .setBiased(0).setPadZeros(0);
    int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f, 13, 5,
                    17, 19, -3,
                    2, -4, 7 };
    float errors[] = { 7.0f };
    float expectedOutput[] = { -7 * 3, - 7 * 13, - 7 * 5, // -21 -91, -35
                                -7 * 17, - 7 * 19, 7 * 3,   // -119, 133, 21
                                - 7 * 2,  7 * 4, - 7 * 7 }; // -14, 28, -49

    testBackpropWeights(dim, batchSize, learningMultiplier, data, errors, expectedOutput);
}

TEST(testupdateweights, backprop_weights_2_upstreamimagesize4_filtersize3) {
    LayerDimensions dim;
    dim.setInputSize(4).setInputPlanes(1).setNumFilters(1).setFilterSize(3)
        .setBiased(0).setPadZeros(0);
    int batchSize = 1;
    const float learningMultiplier = 1;

    float data[] = { 3.0f, 13, 5, 8,
                    17, 19, -3, 2,
                    2, -4, 7, 0,
                    0, 6, 8, 9 };
    float errors[] = { 7.0f, 2,
                        0, -3 };
    float expectedOutput[] = { -3*7-13*2-0+19*3, -999, -999 , // 10
                                -999, -999, -999,
                                -999, -999, -49+27 };          //           -22

    testBackpropWeights(dim, batchSize, learningMultiplier, data, errors, expectedOutput);
}

TEST(testupdateweights, backprop_weights_2_upstreamimagesize5_filtersize3) {
    LayerDimensions dim;
    dim.setInputSize(5).setInputPlanes(1).setNumFilters(1).setFilterSize(3)
        .setBiased(0).setPadZeros(0);
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
    float expectedOutput[] = { -(3*7+13*2-1*5+0*17-3*19-1*3+2*2+1*4+0*7), -999, -999 , // 10
                                -999, -(19*7-3*2-2*1+  0-3*7+0*1   +2*6-1*8+0), -999,
                                -999, -999, -(7*7+0+2*1   +0-3*9+1*4   +5*2-1*3+0) };          //           -22
    testBackpropWeights(dim, batchSize, learningMultiplier, data, errors, expectedOutput);
}

float *allocateInputCleared(int batchSize, LayerDimensions &dim) {
    int inputNumElements = batchSize * dim.inputCubeSize;
    float *data = new float[ inputNumElements ];
    memset(data, 0, sizeof(float) * inputNumElements);
    return data;
}

float *allocateErrorsCleared(int batchSize, LayerDimensions &dim) {
    int outputNumElements = batchSize * dim.outputCubeSize;
    float *errors = new float[ outputNumElements ];
    memset(errors, 0, sizeof(float) * outputNumElements);
    return errors;
}

TEST(testupdateweights, backprop_weights_2_upstreamimagesize3_filtersize1) {
    LayerDimensions dim;
    dim.setInputSize(3).setInputPlanes(1).setNumFilters(1).setFilterSize(1)
        .setBiased(0).setPadZeros(0);
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared(batchSize, dim);
    data[0] = 2;
    data[1 * dim.inputSize + 1] = 7;
    data[2 * dim.inputSize + 2] = 5;

    float *errors = allocateErrorsCleared(batchSize, dim);
    errors[0] = 5;
    errors[1 * dim.outputSize + 1] = 11;
    errors[2 * dim.outputSize + 2] = 3;

    float expectedOutput[] = { -(2 * 5 +  5 * 3 + 7 * 11) };          //           

    testBackpropWeights(dim, batchSize, learningMultiplier, data, errors, expectedOutput);
}

TEST(testupdateweights, backprop_weights_2_upstreamimagesize16_filtersize1) {
    LayerDimensions dim;
    dim.setInputSize(16).setInputPlanes(1).setNumFilters(1).setFilterSize(1)
        .setBiased(0).setPadZeros(0);
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared(batchSize, dim);
    data[0] = 2;
    data[15 * dim.inputSize + 15] = 5;

    float *errors = allocateErrorsCleared(batchSize, dim);
    errors[0] = 4;
    errors[15 * dim.outputSize + 15] = 3;

    float expectedOutput[] = { -(2 * 4 +  3 * 5) };          //           

    testBackpropWeights(dim, batchSize, learningMultiplier, data, errors, expectedOutput);
}

TEST(testupdateweights, backprop_weights_2_upstreamimagesize17_filtersize1) {
    LayerDimensions dim;
    dim.setInputSize(17).setInputPlanes(1).setNumFilters(1).setFilterSize(1)
        .setBiased(0).setPadZeros(0);
    int batchSize = 1;
    const float learningMultiplier = 1;
    cout << dim << endl;

    float *data = allocateInputCleared(batchSize, dim);
    data[0] = 2;
    data[1] = 3.2f;
    data[2] = 1.234f;
    data[16 * dim.inputSize + 16] = 5;

    float *errors = allocateErrorsCleared(batchSize, dim);
    errors[0] = 4;
    errors[1] = -2.5f;
    errors[2] = 4.125f;
    errors[16 * dim.outputSize + 16] = 3;

    float expectedOutput[] = { -(4*2 - 3.2f * 2.5f + 1.234f * 4.125f + 3*5) };          // 

    testBackpropWeights(dim, batchSize, learningMultiplier, data, errors, expectedOutput);
}

TEST(testupdateweights, backprop_weights_2_upstreamimagesize17_filtersize1_moredata) {
    LayerDimensions dim;
    dim.setInputSize(17).setInputPlanes(1).setNumFilters(1).setFilterSize(1)
        .setBiased(0).setPadZeros(0);
    int batchSize = 1;
    const float learningMultiplier = 1;

    float *data = allocateInputCleared(batchSize, dim);
    for(int i = 0; i < square(dim.inputSize); i++) {
        data[i] = ((1 + i) % 20) / 5.3f;
    }

    float *errors = allocateErrorsCleared(batchSize, dim);
    for(int i = 0; i < square(dim.outputSize); i++) {
        errors[i] = ((2 + i) % 17) / 4.2f;
    }

    float expectedOutput[1];
    expectedOutput[0] = 0;
    for (int i = 0; i < square(dim.inputSize); i++) {
        expectedOutput[0] += - data[i] * errors[i];
    }
    cout << "expectedresult: " << expectedOutput[0] << endl;

    testBackpropWeights(dim, batchSize, learningMultiplier, data, errors, expectedOutput);
}

TEST(testupdateweights, backprop_instance3_smaller2) {
    LayerDimensions dim;
    dim.setInputSize(96).setInputPlanes(1).setNumFilters(1).setFilterSize(6)
        .setBiased(0).setPadZeros(0);
    int batchSize = 1;
//    const float learningRate = 1;

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    int outputNumElements = batchSize * dim.outputCubeSize;
    int inputNumElements = batchSize * dim.inputCubeSize;
    int weightsSize = dim.filtersSize;
//    int biasSize = dim.numFilters;

    cout << "numweights: " << weightsSize << endl;

    float *errors = new float[max(10000, outputNumElements)];
    float *inputData = new float[max(10000, inputNumElements)];
    float *weights0 = new float[max(10000, weightsSize) ];
    float *weights1 = new float[max(10000, weightsSize) ];

    memset(errors, 0, sizeof(float) * max(10000, outputNumElements));
    memset(inputData, 0, sizeof(float) * max(10000, inputNumElements));
    memset(weights0, 0, sizeof(float) * max(10000, weightsSize));
    memset(weights1, 0, sizeof(float) * max(10000, weightsSize));

    CLWrapper *errorsWrap = cl->wrap(10000, errors);
    CLWrapper *inputWrap = cl->wrap(10000, inputData);
    CLWrapper *weights0Wrap = cl->wrap(10000, weights0);
    CLWrapper *weights1Wrap = cl->wrap(10000, weights1);

    for(int i = 0 * dim.inputSize; i < dim.inputSize * dim.inputSize; i+= dim.inputSize * 4) {
        inputData[i] = 3;
    }

    for(int i = 0; i < dim.outputSize * dim.outputSize; i+= dim.outputSize) {
        errors[i] = 2;
    }

    errorsWrap->copyToDevice();
    inputWrap->copyToDevice();
    weights0Wrap->copyToDevice();
    weights1Wrap->copyToDevice();
    
    BackpropWeights *backpropWeightsImpl0 = BackpropWeights::instanceSpecific(0, cl, dim);
    backpropWeightsImpl0->debug = true;
    backpropWeightsImpl0->calcGradWeights(batchSize, errorsWrap, inputWrap, weights0Wrap, 0);
    BackpropWeights *backpropWeightsImpl1 = BackpropWeights::instanceSpecific(3, cl, dim);
    backpropWeightsImpl1->debug = true;
    backpropWeightsImpl1->calcGradWeights(batchSize, errorsWrap, inputWrap, weights1Wrap, 0);
    weights0Wrap->copyToHost();
    weights1Wrap->copyToHost();

    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
            cout << weights0[i*6+j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
            cout << weights1[i*6+j] << " ";
        }
        cout << endl;
    }

    cout << endl;
    int isok = 1;
    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
            if(weights0[i*6+j] == weights1[i*6+j]) {
                cout << ".";
            } else {
                cout << "!";
                isok = 0;
            }
        }
        cout << endl;
    }
    cout << endl;
    EXPECT_EQ(1, isok);

    for(int i = 0; i < 12; i++) {
        cout << i << "=";
        for(int slice = 0; slice < 8; slice++) {
            cout << weights1[100+ 12 * slice + i] << " ";
        }
        cout << endl;
    }
    cout << endl;

    for(int i = 0; i < 20; i++) {
        cout << i << "=";
        for(int slice = 0; slice < 8; slice++) {
            cout << weights1[200+ 20 * slice + i] << " ";
        }
        cout << endl;
    }
    delete backpropWeightsImpl0;
    delete backpropWeightsImpl1;
    delete errorsWrap;
    delete inputWrap;
    delete weights0Wrap;
    delete weights1Wrap;
    delete[] errors;
    delete[] inputData;
    delete[] weights0;
    delete[] weights1;
    delete cl;
}

class CompareSpecificArgs {
public:
    static CompareSpecificArgs instance(){ CompareSpecificArgs args; return args; };

    // [[[cog
    // floats= []
    // ints = [  'inputPlanes', 'inputSize', 'numFilters', 'filterSize',
    //    'batchSize', 'biased', 'padZeros', 'instance0', 'instance1' ]
    // import cog_fluent
    // cog_fluent.gov3('CompareSpecificArgs', ints = ints, floats = floats)
    // ]]]
    // generated, using cog:
    int _inputPlanes;
    int _inputSize;
    int _numFilters;
    int _filterSize;
    int _batchSize;
    int _biased;
    int _padZeros;
    int _instance0;
    int _instance1;
    CompareSpecificArgs() {
        _inputPlanes = 0;
        _inputSize = 0;
        _numFilters = 0;
        _filterSize = 0;
        _batchSize = 0;
        _biased = 0;
        _padZeros = 0;
        _instance0 = 0;
        _instance1 = 0;
    }
    CompareSpecificArgs inputPlanes( int _inputPlanes ) {
        this->_inputPlanes = _inputPlanes;
        return *this;
    }
    CompareSpecificArgs inputSize( int _inputSize ) {
        this->_inputSize = _inputSize;
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

namespace testupdateweights {

void compareSpecific(bool debug, float learningRate, int its, int batchSize, LayerDimensions dim, int instance0, int instance1) {
    cout << dim << endl;

    int outputNumElements = batchSize * dim.outputCubeSize;
    int inputNumElements = batchSize * dim.inputCubeSize;
    int weightsSize = dim.filtersSize;
    int biasSize = dim.numFilters;

    int outputAllocated = max(10000, outputNumElements);
    int inputAllocated = max(10000, inputNumElements);
    int weightsAllocated = max(10000, weightsSize);
    int biasAllocated = max(10000, biasSize);

//    cout << "numweights: " << weightsSize << endl;

    float *bias1 = new float[ biasAllocated ];
    float *bias2 = new float[ biasAllocated ];
    memset(bias1, 0, sizeof(float) * biasAllocated);
    memset(bias2, 0, sizeof(float) * biasAllocated);

    float *gradOutput = new float[outputAllocated];
    float *inputData = new float[inputAllocated];
    float *weights1 = new float[weightsAllocated];
    float *weights2 = new float[weightsAllocated];

    memset(gradOutput, 0, sizeof(float) * outputAllocated);
    memset(inputData, 0, sizeof(float) * inputAllocated);
    memset(weights1, 0, sizeof(float) * weightsAllocated);
    memset(weights2, 0, sizeof(float) * weightsAllocated);

    WeightRandomizer::randomize(gradOutput, outputAllocated, -0.1f, 0.1f);
    WeightRandomizer::randomize(inputData, inputAllocated, -0.3f, 0.7f);

//    WeightRandomizer::randomizeInts(errors, outputAllocated, 0, 99);
//    WeightRandomizer::randomizeInts(inputData, inputAllocated, 0, 99);

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    
    int instances[2];
    instances[0] = instance0;
    instances[1] = instance1;
    float *weightsByInstance[2];
    weightsByInstance[0] = weights1;
    weightsByInstance[1] = weights2;
    float *biasByInstance[2];
    biasByInstance[0] = bias1;
    biasByInstance[1] = bias2;
    BackpropWeights *instanceObjects[2];
    instanceObjects[0] = BackpropWeights::instanceSpecific(instance0, cl, dim);
    instanceObjects[1] = BackpropWeights::instanceSpecific(instance1, cl, dim);
    for(int instance = 0; instance < 2; instance++) {
        Timer timer;
        BackpropWeights *backpropWeightsImpl = instanceObjects[instance];
        backpropWeightsImpl->debug = true;
        for(int it = 0; it < its; it++) {
            backpropWeightsImpl->calcGradWeights(batchSize,
                gradOutput, inputData, weightsByInstance[instance], biasByInstance[instance]);
        }
        timer.timeCheck("instance " + toString(instances[instance]) + " backpropweights");
//        delete backpropWeightsImpl;
    }
    delete instanceObjects[0];
    delete instanceObjects[1];
    cout << dim << endl;
    for(int i = 0; i < 25; i++) {
        cout << "weights[" << i << "]=" << weights1[i] << " " << weights2[i];
        if(i < weightsSize) {
            if(abs(weights1[i] - weights2[i]) <= abs(weights1[i]) / 10000.0f) {
                if(debug) cout << " SAME";
            } else {
                cout << " DIFF";
            }
        } else {
            if(debug) cout << "     ";
        }
        if(debug) cout << "  || " << weights2[100+i] ;
        if(debug) cout << "  || " << weights2[200+i] ;
        if(debug) cout << "  || " << weights2[300+i] ;
        if(debug) cout << "  || " << weights2[400+i] ;
        if(debug) cout << "  || " << weights2[500+i] ;
        if(debug) cout << "  || " << weights2[600+i] ;
        if(debug) cout << "  || " << weights2[700+i];
        cout << endl;
    }
    bool same = true;
    int errCount = 0;
    for(int i = 0; i < weightsSize; i++) {
        if(abs(weights1[i] - weights2[i]) > 0.001 * max(abs(weights1[i]), abs(weights2[i]))) {
//        if(abs(weights1[i] - weights2[i]) > abs(weights1[i]) / 10000.0f) {
            cout << "DIFF: weights i " << i << " " << weights1[i] << " != " << weights2[i] << endl;
            same = false;
            errCount++;
            if(errCount == 5) {
                cout << " ... " << endl;
                break;
            }
        }
    }
    if(dim.biased) {
        errCount = 0;
        for(int i = 0; i < biasSize; i++) {
            if(abs(bias1[i] - bias2[i]) > 0.001 * max(abs(bias1[i]), abs(bias2[i]))) {
    //        if(abs(weights1[i] - weights2[i]) > abs(weights1[i]) / 10000.0f) {
                cout << "DIFF: bias i " << i << " " << bias1[i] << " != " << bias2[i] << endl;
                same = false;
                errCount++;
                if(errCount == 5) {
                    cout << " etc ... " << endl;
                    break;
                }
            }
        }
    }
    EXPECT_EQ(true, same);

//    delete backpropWeightsImpl1;
//    delete backpropWeightsImpl2;

    delete[] weights1;
    delete[] weights2;
    delete[] gradOutput;
    delete[] inputData;

    delete cl;
}

TEST(SLOW_testupdateweights, compare_args) {
    bool debug = false;
    int instance0 = 1;
    int instance1 = 3;
    LayerDimensions dim;
    dim.setInputSize(28).setInputPlanes(4).setNumFilters(8).setFilterSize(5)
        .setBiased(1).setPadZeros(1);
    int batchSize = 4;
    int its = 1;
//        string activationName = "tanh";
    float learningRate = 1.0f;

    DimFromArgs::arg(&dim);
    TestArgsParser::arg("debug", &debug);
    TestArgsParser::arg("instance0", &instance0);
    TestArgsParser::arg("instance1", &instance1);
    TestArgsParser::arg("its", &its);
    TestArgsParser::arg("batchsize", &batchSize);
//        TestArgsParser::arg("activation", &activationName);
    TestArgsParser::arg("learningrate", &learningRate);
    TestArgsParser::go();
    dim.deriveOthers();
//        ActivationFunction *fn = ActivationFunction::fromName(activationName);

    compareSpecific(debug, learningRate, its, batchSize, dim, instance0, instance1);        
}

//    TEST(testupdateweights, compare_instance3_smaller2) {
//        LayerDimensions dim;
//        dim.setInputSize(96).setInputPlanes(1).setNumFilters(1).setFilterSize(6)
//            .setBiased(0).setPadZeros(0);
//        int batchSize = 1;
//        const float learningRate = 1;
//        compareSpecific(CompareSpecificArgs::instance()
//            .batchSize(1).inputPlanes(1).inputSize(96).numFilters(1)
//            .filterSize(6).biased(0).padZeros(false)
//            .instance0(0).instance1(3));
//    }

//    TEST(SLOW_testupdateweights, compare_specific) {
//        compareSpecific(CompareSpecificArgs::instance()
//            .batchSize(128).inputPlanes(32).inputSize(19).numFilters(32)
//            .filterSize(3).biased(0).padZeros(false)
//            .instance0(1).instance1(3));
//    }

//    TEST(SLOW_testupdateweights, compare_specific_96image) {
//        compareSpecific(CompareSpecificArgs::instance()
//            .batchSize(128).inputPlanes(2).inputSize(96).numFilters(8)
//            .filterSize(6).biased(1).padZeros(false)
//            .instance0(0).instance1(3));
//    }

//    TEST(SLOW_testupdateweights, compare_specific_96image_smaller) {
//        compareSpecific(CompareSpecificArgs::instance()
//            .batchSize(1).inputPlanes(1).inputSize(48).numFilters(1)
//            .filterSize(2).biased(1).padZeros(false)
//            .instance0(0).instance1(3));
//    }

//    TEST(SLOW_testupdateweights, compare_specific_96image_smaller2) {
//        compareSpecific(CompareSpecificArgs::instance()
//            .batchSize(1).inputPlanes(1).inputSize(96).numFilters(1)
//            .filterSize(4).biased(0).padZeros(false)
//            .instance0(0).instance1(3));
//    }

//    TEST(SLOW_testupdateweights, compare_specific_96image_smaller3) {
//        compareSpecific(CompareSpecificArgs::instance()
//            .batchSize(1).inputPlanes(1).inputSize(96).numFilters(1)
//            .filterSize(6).biased(false).padZeros(false)
//            .instance0(0).instance1(3));
//    }

//    TEST(SLOW_testupdateweights, compare_specific_96image_smaller4) {
//        compareSpecific(CompareSpecificArgs::instance()
//            .batchSize(1).inputPlanes(2).inputSize(96).numFilters(8)
//            .filterSize(4).biased(1).padZeros(false)
//            .instance0(0).instance1(3));
//    }

void measurePerf(int batchSize, LayerDimensions dim, int instance) {

    int outputNumElements = batchSize * dim.outputCubeSize;
    int inputNumElements = batchSize * dim.inputCubeSize;
    int weightsSize = dim.filtersSize;
    int biasSize = dim.numFilters;

    int outputAllocated = outputNumElements;
    int inputAllocated = inputNumElements;
    int weightsAllocated = weightsSize;
    int biasAllocated = biasSize;

    cout << "numweights: " << weightsSize << endl;

    float *bias = new float[ biasAllocated ];
    memset(bias, 0, sizeof(float) * biasAllocated);

    float *gradOutput = new float[outputAllocated];
    float *inputData = new float[inputAllocated];
    float *weights = new float[weightsAllocated];

    memset(gradOutput, 0, sizeof(float) * outputAllocated);
    memset(inputData, 0, sizeof(float) * inputAllocated);
    memset(weights, 0, sizeof(float) * weightsAllocated);

    WeightRandomizer::randomizeInts(gradOutput, outputAllocated, 0, 99);
    WeightRandomizer::randomizeInts(inputData, inputAllocated, 0, 99);

    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    
    BackpropWeights *backpropWeightsImpl = BackpropWeights::instanceSpecific(instance, cl, dim);
    Timer timer;
    backpropWeightsImpl->calcGradWeights(batchSize, gradOutput, inputData, weights, bias);
    timer.timeCheck("backprop time");

    delete backpropWeightsImpl;

    delete[] weights;
    delete[] gradOutput;
    delete[] inputData;

    delete cl;
}

}

