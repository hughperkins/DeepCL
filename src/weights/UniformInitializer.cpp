// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/RandomSingleton.h"
#include "weights/UniformInitializer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

UniformInitializer::UniformInitializer(float multiplier) {
    this->multiplier = multiplier;
}
VIRTUAL void UniformInitializer::initializeWeights(int numWeights, float *weights, int fanin) {
    float range = multiplier / (float)fanin;
    for(int i = 0; i < numWeights; i++) {
        float uniformrand = RandomSingleton::uniform() * 2.0f - 1.0f;  
        weights[i] = range * uniformrand;
    }
}
VIRTUAL void UniformInitializer::initializeBias(int numBias, float *bias, int fanin) {
    float range = multiplier / (float)fanin;
    for(int i = 0; i < numBias; i++) {
        float uniformrand = RandomSingleton::uniform() * 2.0f - 1.0f;  
        bias[i] = range * uniformrand;
    }
}

