// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include "Layer.h"
#include "NeuralNet.h"
#include "BatchLearner.h"
#include "NetAction.h"
#include "EpochMaker.h"

using namespace std;

float EpochMaker::run() {
    if( _labels != 0 ) {
        throw runtime_error("should not provide labels if using Epoch::run");
    }
    if( _expectedOutputs == 0 ) {
        throw runtime_error("must provide expectedOutputs if using runWithCalcTrainingAccuracy");
    }
    BatchLearner batchLearner( net );
    return batchLearner.runEpochFromExpected( trainer, _batchSize, _numExamples, _inputData, _expectedOutputs );
}

//float EpochMaker::runWithCalcTrainingAccuracy( int *p_numRight ) {
//    if( _expectedOutputs == 0 ) {
//        throw runtime_error("must provide expectedOutputs if using Epoch::runWithCalcTrainingAccuracy");
//    }
//    if( _expectedOutputs == 0 ) {
//        throw runtime_error("must provide labels if using Epoch::runWithCalcTrainingAccuracy");
//    }
//    return net->doEpochWithCalcTrainingAccuracy( _learningRate, _batchSize, _numExamples, _inputData, _expectedOutputs, _labels, p_numRight );
//}

//float EpochMaker::runFromLabels( int *p_numRight ) {
//    if( _expectedOutputs != 0 ) {
//        throw runtime_error("should not provide expectedOutputs if using Epoch::runFromLabels");
//    }
//    if( _labels == 0 ) {
//        throw runtime_error("must provide labels if using Epoch::runFromLabels");
//    }
//    BatchLearner batchLearner( net );
//    EpochResult epochResult = batchLearner.runEpochFromLabels( _learningRate, _batchSize, _numExamples, _inputData, _labels );
//    *p_numRight = epochResult.numRight;
//    return epochResult.loss;
//}


