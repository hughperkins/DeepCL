#pragma once

#include <cstring>

class NeuralNet;

class EpochMaker {
    NeuralNet *net;
    float _learningRate;
    int _batchSize;
    int _numExamples;
    float *_inputData;
    float *_expectedOutputs;
public:
    EpochMaker( NeuralNet *net ) {
        memset( this, 0, sizeof(EpochMaker) );
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
    float run();
    float runWithCalcTrainingAccuracy(int *trainingLabels, int *p_numRight);
};

