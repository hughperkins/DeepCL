// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/RandomSingleton.h"
#include "weights/OriginalInitializer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL void OriginalInitializer::initializeWeights(int numWeights, float *weights, int fanin) {
    float rangesize = sqrt(12.0f / (float)fanin) ;
    for(int i = 0; i < numWeights; i++) {
        float uniformrand = RandomSingleton::uniform();  
        float weight = rangesize * (uniformrand - 0.5f);
        weights[i] = weight;
    }
}
VIRTUAL void OriginalInitializer::initializeBias(int numBias, float *bias, int fanin) {
    float rangesize = sqrt(12.0f / (float)fanin) ;
    for(int i = 0; i < numBias; i++) {
        float uniformrand = RandomSingleton::uniform();  
        float weight = rangesize * (uniformrand - 0.5f);
        bias[i] = weight;
    }
}

