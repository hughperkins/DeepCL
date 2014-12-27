#include "EpochMaker.h"

#include "NeuralNet.h"

void EpochMaker::run() {
    net->doEpoch( _learningRate, _batchSize, _numExamples, _inputData, _expectedOutputs );
}


