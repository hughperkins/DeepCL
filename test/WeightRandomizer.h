#pragma once

#include <random>

class WeightRandomizer {
public:
    static void randomize( float *values, int numValues ) {
        std::mt19937 random;
        random.seed(0); // so always gives same results
        for( int i = 0; i < numValues; i++ ) {
            values[i] = random() / (float)std::mt19937::max() * 0.2f - 0.1f;
        }
    }
};

