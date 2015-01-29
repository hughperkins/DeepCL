// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationHelper.h"
#include "NeuralNet.h"

#include "BatchLearner.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

BatchLearner::BatchLearner( NeuralNet *net, float dataTranslate, float dataScale ) :
    net( net ),
    dataTranslate( dataTranslate ),
    dataScale( dataScale ) {
}

VIRTUAL float BatchLearner::getLoss() const {
    return loss;
}

VIRTUAL int BatchLearner::getNumRight() const {
    return numRight;
}



