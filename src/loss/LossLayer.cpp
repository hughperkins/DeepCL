// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "loss/LossLayer.h"
#include "loss/IAcceptsLabels.h"
#include "batch/BatchData.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

LossLayer::LossLayer(Layer *previousLayer, LossLayerMaker *maker) :
        Layer(previousLayer, maker) {
}
VIRTUAL void LossLayer::forward() {
}
VIRTUAL bool LossLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL float *LossLayer::getOutput() {
    return previousLayer->getOutput();
}
VIRTUAL int LossLayer::getOutputNumElements() const {
    return previousLayer->getOutputNumElements();
}
VIRTUAL int LossLayer::getOutputCubeSize() const {
    return previousLayer->getOutputCubeSize();
}
VIRTUAL int LossLayer::getOutputSize() const {
    return previousLayer->getOutputSize();
}
VIRTUAL int LossLayer::getOutputPlanes() const {
    return previousLayer->getOutputPlanes();
}
VIRTUAL int LossLayer::getWeightsSize() const {
    return previousLayer->getWeightsSize();
}

VIRTUAL float LossLayer::calcLoss(OutputData *outputData) {
    ExpectedData *expectedData = dynamic_cast< ExpectedData * >(outputData);
    LabeledData *labeledData = dynamic_cast< LabeledData * >(outputData);
    if(expectedData != 0) {
        return this->calcLoss(expectedData->expected);
    } else if(labeledData != 0) {
        IAcceptsLabels *labeled = dynamic_cast< IAcceptsLabels * >(this);
        return labeled->calcLossFromLabels(labeledData->labels);
    } else {
        throw runtime_error("OutputData child class not implemeneted in LossLayer::calcLoss");
    }
}

VIRTUAL void LossLayer::calcGradInput(OutputData *outputData) {
    ExpectedData *expectedData = dynamic_cast< ExpectedData * >(outputData);
    LabeledData *labeledData = dynamic_cast< LabeledData * >(outputData);
    if(expectedData != 0) {
        this->calcGradInput(expectedData->expected);
    } else if(labeledData != 0) {
        IAcceptsLabels *labeled = dynamic_cast< IAcceptsLabels * >(this);
        labeled->calcGradInputFromLabels(labeledData->labels);
    } else {
        throw runtime_error("OutputData child class not implemeneted in LossLayer::calcGradInput");
    }
}

VIRTUAL int LossLayer::calcNumRight(OutputData *outputData) {
    ExpectedData *expectedData = dynamic_cast< ExpectedData * >(outputData);
    LabeledData *labeledData = dynamic_cast< LabeledData * >(outputData);
    if(expectedData != 0) {
        return 0; // how are we going to calculate num right, if not labeled?
    } else if(labeledData != 0) {
        IAcceptsLabels *labeled = dynamic_cast< IAcceptsLabels * >(this);
        return labeled->calcNumRightFromLabels(labeledData->labels);
    } else {
        throw runtime_error("OutputData child class not implemeneted in LossLayer::calcNumRight");
    }
}

