#include "EpochMaker.h"

#include "NeuralNet.h"

#include <stdexcept>
using namespace std;

float EpochMaker::run() {
    return net->doEpoch( _learningRate, _batchSize, _numExamples, _inputData, _expectedOutputs );
}


