// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/stringhelper.h"
#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "loss/LossLayer.h"
#include "trainers/SGDStateMaker.h"
#include "trainers/SGDState.h"
#include "trainers/SGD.h"
#include "loss/IAcceptsLabels.h"
#include "batch/NetAction.h"
#include "clmath/CLMathWrapper.h"
#include "batch/BatchData.h"

using namespace std;
using namespace easycl;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


VIRTUAL SGD::~SGD() {
}
VIRTUAL void SGD::setMomentum(float momentum) {
    this->momentum = momentum;
}
VIRTUAL void SGD::setWeightDecay(float weightDecay) {
    this->weightDecay = weightDecay;
}
VIRTUAL std::string SGD::asString() {
    return "SGD{ learningRate=" + toString(learningRate) + ", momentum=" + 
        toString(momentum) + " }";
}
VIRTUAL void SGD::updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
        SGDState *trainerState) {
    int numWeights = trainerState->numWeights;
    CLWrapper *lastUpdateWrapper = trainerState->lastUpdateWrapper;
    float *gradWeightsCopy = new float[ numWeights ];
    CLWrapper *gradWeightsCopyWrapper = cl->wrap(numWeights, gradWeightsCopy);
    gradWeightsCopyWrapper->createOnDevice();

    CLMathWrapper lastUpdates_(lastUpdateWrapper);
    CLMathWrapper gradWeights_(gradWeightsWrapper);
    CLMathWrapper gradWeightsCopy_(gradWeightsCopyWrapper);
    CLMathWrapper weights_(weightsWrapper);

    // following all happens on gpu, via clmathwrapper:
    lastUpdates_ *= momentum;
    gradWeightsCopy_ = gradWeights_;
    gradWeightsCopy_ *= - learningRate;
    lastUpdates_ += gradWeightsCopy_;
    weights_ += lastUpdates_;

    if(weightDecay > 0) {
        // apply weight decay, by multiplying the weights by (1.0f - weightDecay)
        // so weightDecay == 0 means no decay; and weightDecay == 1.0f means
        // weights go immediately to zero
        weights_ *= 1.0f - weightDecay;
    }

    delete gradWeightsCopyWrapper;
    delete[] gradWeightsCopy;
}
VIRTUAL BatchResult SGD::trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, OutputData *outputData) {
    // learns one batch, including updating weights
    // doesnt have to think about running multiple batches,
    // or loading data, or anything like that
    bindState(net);

    net->forward(input);
    int numRight = net->calcNumRight(outputData);
    float loss = net->calcLoss(outputData);
    net->backward(outputData);

    int numLayers = net->getNumLayers();
    for(int layerIdx = numLayers - 2; layerIdx > 0; layerIdx--) {
        Layer *layer = net->getLayer(layerIdx);
        if(!layer->needsBackProp()) {
            break;
        }
        if(layer->needsTrainerState()) {
            updateWeights(layer->getWeightsWrapper(), layer->getGradWeightsWrapper(), 
                dynamic_cast< SGDState * >(layer->getTrainerState()) );
            if(layer->biased()) {
                updateWeights(layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< SGDState * >(layer->getBiasTrainerState()) );
            }
        }
    }
    return BatchResult(loss, numRight);
}
VIRTUAL BatchResult SGD::trainNet(NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput) {
    ExpectedData expectedData(net, expectedOutput);
    return this->trainNet(net, context, input, &expectedData);
}
VIRTUAL BatchResult SGD::trainNetFromLabels(NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels) {
    LabeledData labeledData(net, labels);
    return this->trainNet(net, context, input, &labeledData);
}
VIRTUAL void SGD::bindState(NeuralNet *net) {
    SGDStateMaker stateMaker;
    this->_bindState(net, &stateMaker);
}
STATIC SGD *SGD::instance(EasyCL *cl, float learningRate) {
    SGD *sgd = new SGD(cl);
    sgd->setLearningRate(learningRate);
    return sgd;
}
STATIC SGD *SGD::instance(EasyCL *cl, float learningRate, float momentum) {
    SGD *sgd = new SGD(cl);
    sgd->setLearningRate(learningRate);
    sgd->setMomentum(momentum);
    return sgd;
}
SGD::SGD(EasyCL *cl) :
        Trainer(cl),
        momentum(0.0f),
        weightDecay(0.0f) {
}

