#include "EpochMaker.h"

#include "NeuralNet.h"

#include <stdexcept>
using namespace std;

void EpochMaker::run() {
    net->doEpoch( _learningRate, _batchSize, _numExamples, _inputData, _expectedOutputs );
}


