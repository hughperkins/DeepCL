// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <random>

#include "Trainable.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

void Trainable::learnBatch( float learningRate, float const*images, float const *expectedResults ) {
    setTraining( true );
    propagate( images);
    backProp( learningRate, expectedResults );
}
void Trainable::learnBatch( float learningRate, unsigned char const*images, float const *expectedResults ) {
    setTraining( true );
    propagate( images);
    backProp( learningRate, expectedResults );
}
void Trainable::learnBatchFromLabels( float learningRate, float const*images, int const *labels ) {
    setTraining( true );
    propagate( images);
    backPropFromLabels( learningRate, labels );
}
void Trainable::learnBatchFromLabels( float learningRate, unsigned char const*images, int const *labels ) {
    setTraining( true );
    propagate( images);
    backPropFromLabels( learningRate, labels );
}

