// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <random>
#include <algorithm>

#include "NeuralNet.h"
#include "Backward.h"
#include "ActivationFunction.h"
#include "LossLayer.h"
#include "ForceBackpropLayerMaker.h"
#include "ActivationMaker.h"

#include "gtest/gtest.h"

#include "test/gtest_supp.h"
#include "test/Sampler.h"
#include "test/WeightRandomizer.h"

#include "SGD.h"

using namespace std;

TEST( testsgd, basic ) {
    // this is mostly to help me figure out how to design the SGD and Trainer classes
    NeuralNet *net = new NeuralNet( 1, 5 );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(1)->filterSize(3)->biased(0)->padZeros(0) );
    net->addLayer( SquareLossMaker::instance() );
    cout << net->asString() << endl;
    net->setBatchSize(2);
    OpenCLHelper *cl = net->getCl();

    int batchSize = dynamic_cast< InputLayer *>(net->getLayer(0))->batchSize;
//    const int outputPlanes = net->getOutputPlanes();

    int inputCubeSize = net->getInputCubeSize();
    int outputCubeSize = net->getOutputCubeSize();

    int inputTotalSize = inputCubeSize * batchSize;
    int outputTotalSize = outputCubeSize * batchSize;

    cout << "inputtotalsize=" << inputTotalSize << " outputTotalSize=" << outputTotalSize << endl;

    float *input = new float[inputTotalSize];
    float *expectedOutput = new float[outputTotalSize];

    WeightRandomizer::randomize( 0, input, inputTotalSize, 0.0f, 1.0f );
    WeightRandomizer::randomize( 1, expectedOutput, outputTotalSize, 0.0f, 1.0f );

    SGD *sgd = new SGD( cl, net );
    sgd->setLearningRate( 0.002f );
    sgd->setMomentum( 0.1f );

//    net->forward( input );
    sgd->learn( input, expectedOutput );

    delete sgd;
    delete net;
}

