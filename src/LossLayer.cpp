// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "LossLayer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

LossLayer::LossLayer( Layer *previousLayer, LossLayerMaker const*maker ) :
        Layer( previousLayer, maker ) {
}
VIRTUAL void LossLayer::propagate() {
}
VIRTUAL bool LossLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL float *LossLayer::getResults() {
    return previousLayer->getResults();
}
VIRTUAL int LossLayer::getResultsSize() const {
    return previousLayer->getResultsSize();
}
VIRTUAL int LossLayer::getOutputCubeSize() const {
    return previousLayer->getOutputCubeSize();
}
VIRTUAL int LossLayer::getOutputBoardSize() const {
    return previousLayer->getOutputBoardSize();
}
VIRTUAL int LossLayer::getOutputPlanes() const {
    return previousLayer->getOutputPlanes();
}
VIRTUAL int LossLayer::getWeightsSize() const {
    return previousLayer->getWeightsSize();
}

