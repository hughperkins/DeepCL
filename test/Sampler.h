#pragma once

#include <random>
#include <string>
#include <iostream>

#include "layer/Layer.h"
#include "util/RandomSingleton.h"

class Sampler {
public:
    static void printSamples( std::string arrayName, int arraySize, float *array, int numSamples = 5 ) {
        std::mt19937 random;
        random.seed(1);
        for( int sample = 0; sample < numSamples; sample++ ) {
            int index = random() % arraySize;
            std::cout << "EXPECT_FLOAT_NEAR( " << array[index] << ", " << arrayName << "[" << index << "] );" << std::endl;
        }
    }
    static void sampleWeights( std::string label, Layer *layer ) {
        float *weights = layer->getWeights();
        int numWeights = layer->getWeightsSize();
        MT19937 random;
        random.seed(1);
//        RandomSingleton *random = RandomSingleton::instance();
//        int sampleIdxs[5];
        for( int sample = 0; sample < 5; sample++ ) {
            int sampleIdx = random() % ( numWeights - 1 );
//            sampleIdxs[sample] = sampleIdx;
            std::cout << label << ": weights[" << sampleIdx << "]=" << weights[sampleIdx] << std::endl;
        }
    }
    static void sampleFloatWrapper( std::string label, CLWrapper *wrapper ) {
        int N = wrapper->size();
        MT19937 random;
        random.seed(1);
        wrapper->copyToHost();
        float *hostArray = (float *)wrapper->getHostArray();
        for( int sample = 0; sample < 5; sample++ ) {
            int idx = random() % N;
            std::cout << label + ": sample[" << idx << "]=" << hostArray[idx] << std::endl;
        }
    }
    static void sampleFloats( std::string label, int N, float *floats ) {
        MT19937 random;
        random.seed(1);
        for( int sample = 0; sample < 5; sample++ ) {
            int idx = random() % N;
            std::cout << label + ": sample[" << idx << "]=" << floats[idx] << std::endl;
        }
    }
};


