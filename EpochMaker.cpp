#include "EpochMaker.h"

#include "NeuralNet.h"

#include <stdexcept>
using namespace std;

float EpochMaker::run() {
    return net->doEpoch( _learningRate, _batchSize, _numExamples, _inputData, _expectedOutputs );
}

float EpochMaker::runWithCalcTrainingAccuracy( int *trainingLabels, int *p_numRight ) {
    return net->doEpochWithCalcTrainingAccuracy( _learningRate, _batchSize, _numExamples, _inputData, _expectedOutputs, trainingLabels, p_numRight );
}


