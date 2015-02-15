#pragma once

#include <cstring>

#include "ClConvolveDllExport.h"

class NeuralNet;

class ClConvolve_EXPORT EpochMaker {
    NeuralNet *net;
    float _learningRate;
    int _batchSize;
    int _numExamples;
    float *_inputData;
    float *_expectedOutputs;
    int const*_labels;
public:
    EpochMaker( NeuralNet *net ) {
        memset( this, 0, sizeof(EpochMaker) );
        _expectedOutputs = 0;
        _labels = 0;
        this->net = net;
    }
    EpochMaker *learningRate(float learningRate){
        this->_learningRate = learningRate;
        return this;
    }
    EpochMaker *batchSize(int batchSize){
        this->_batchSize = batchSize;
        return this;
    }
    EpochMaker *numExamples(int numExamples){
        this->_numExamples = numExamples;
        return this;
    }
    EpochMaker *inputData(float *inputData){
        this->_inputData = inputData;
        return this;
    }
    EpochMaker *expectedOutputs(float *expectedOutputs){
        this->_expectedOutputs = expectedOutputs;
        return this;
    }
    EpochMaker *labels(int const*labels){
        this->_labels = labels;
        return this;
    }
    float run();
    float runWithCalcTrainingAccuracy( int *p_numRight);
    float runFromLabels( int *p_numRight);
};

