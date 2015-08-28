// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <random>

#include "net/Trainable.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

//void Trainable::learnBatch(float learningRate, float const*images, float const *expectedOutput) {
//    setTraining(true);
//    forward(images);
//    backward(learningRate, expectedOutput);
//}
//void Trainable::learnBatchFromLabels(float learningRate, float const*images, int const *labels) {
//    setTraining(true);
//    forward(images);
//    backwardFromLabels(learningRate, labels);
//}

