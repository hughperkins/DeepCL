// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "SGDStateMaker.h"
#include "SGDState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

SGDStateMaker::SGDStateMaker() {
//    this->learningRate = learningRate;
//    this->momentum = 0.0f;
}

//SGDStateMaker::SGDStateMaker( float learningRate ) {
//    this->learningRate = learningRate;
//    this->momentum = 0.0f;
//}

//SGDStateMaker::SGDStateMaker( float learningRate, float momentum ) {
//    this->learningRate = learningRate;
//    this->momentum = momentum;
//}

TrainerState *SGDStateMaker::instance( EasyCL *cl, int numWeights ) {
    SGDState *sgd = new SGDState( cl, numWeights );
//    sgd->learningRate = learningRate;
//    sgd->momentum = momentum;
    return sgd;
}

