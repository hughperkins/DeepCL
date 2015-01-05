// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include "Layer.h"
#include "NeuralNet.h"

#include "EpochMaker.h"

using namespace std;

float EpochMaker::run() {
    return net->doEpoch( _learningRate, _batchSize, _numExamples, _inputData, _expectedOutputs );
}

float EpochMaker::runWithCalcTrainingAccuracy( int *trainingLabels, int *p_numRight ) {
    return net->doEpochWithCalcTrainingAccuracy( _learningRate, _batchSize, _numExamples, _inputData, _expectedOutputs, trainingLabels, p_numRight );
}


