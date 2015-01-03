#pragma once

#include <random>
#include <string>
#include <iostream>

class Sampler {
public:
    static void printSamples( std::string arrayName, int arraySize, float *array, int numSamples = 5 ) {
        std::mt19937 random;
        random.seed(0);
        for( int sample = 0; sample < numSamples; sample++ ) {
            int index = random() % arraySize;
            std::cout << "EXPECT_FLOAT_NEAR( " << array[index] << ", " << arrayName << "[" << index << "] );" << std::endl;
        }
    }
};


