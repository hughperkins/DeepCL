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
#include "trainers/NesterovStateMaker.h"
#include "trainers/NesterovState.h"
#include "trainers/Nesterov.h"
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


VIRTUAL Nesterov::~Nesterov() {
}
VIRTUAL void Nesterov::setMomentum(float momentum) {
    this->momentum = momentum;
}
VIRTUAL std::string Nesterov::asString() {
    return "Nesterov{ learningRate=" + toString(learningRate) + ", momentum=" + 
        toString(momentum) + " }";
}
VIRTUAL void Nesterov::loadFutureWeights(
        CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
        NesterovState *trainerState) {
    // this will save the old weights, into the trainerState,
    // and then add mom * dweights to them

    // create CLMathWrapper objects, so we can do per-element maths on the gpu:
    CLMathWrapper clOldWeights(trainerState->oldWeightsWrapper);
    CLMathWrapper clWeights(weightsWrapper);
    CLMathWrapper clGradWeights(gradWeightsWrapper);

    // following happens on the gpu:
    clOldWeights = clWeights;
    clWeights = clGradWeights;
    clWeights *= momentum;
    clWeights += clOldWeights;
}
VIRTUAL void Nesterov::updateWeights(CLWrapper *weightsWrapper,
        CLWrapper *gradWeightsWrapper,
        NesterovState *trainerState) {
    // we have: gradWeights = gradient(weights[t] + mom * dweights[t])
    //          trainerState->oldWeights = weights[t]
    //          trainerState->lastUpdate = dweights[t]
    // and so we can calculate
    //      dweights[t+1] = mom * dweights[t] - learningrate * gradient( 
    //                          weights[t] + mom * dweights[t])
    //      weights[t+1] = weights[t] + dweights[t+1]

    // create CLMathWrapper objects, so we can do per-element maths on the gpu:
    CLMathWrapper clLastUpdate(trainerState->lastUpdateWrapper);
    CLMathWrapper clOldWeights(trainerState->oldWeightsWrapper);
    CLMathWrapper clGradWeights(gradWeightsWrapper);
    CLMathWrapper clWeights(weightsWrapper);

    // following happens on the gpu, via CLMathWrapper:

    clGradWeights *= - learningRate;
    clLastUpdate *= momentum;
    clLastUpdate += clGradWeights;
    clWeights = clOldWeights;
    clWeights += clLastUpdate;
}
VIRTUAL BatchResult Nesterov::trainNet( 
    NeuralNet *net, TrainingContext *context,
    float const *input, OutputData *outputData) {
    // learns one batch, including updating weights
    // doesnt have to think about running multiple batches,
    // or loading data, or anything like that

    //      dweights[t+1] = mom * dweights[t] - learningrate * gradient(
    //                      weights[t] + mom * dweights[t])
    //      weights[t+1] = weights[t] + dweights[t+1]
    //
    // given weights[t], dweights[t]:
    //      forward/backprop weights[t] + mom * dweights[t]
    //      => calc dweights[t+1]
    //      => calc weights[t+1]
    bindState(net);

    // first, substitute weights + mom * dweights into the weights
    // calculate them first
    // save old weights first I suppose?

    int numLayers = net->getNumLayers();
    for(int layerIdx = numLayers - 2; layerIdx > 0; layerIdx--) {
        Layer *layer = net->getLayer(layerIdx);
        if(!layer->needsBackProp()) {
            break;
        }
        if(layer->needsTrainerState()) {
            loadFutureWeights(layer->getWeightsWrapper(), layer->getGradWeightsWrapper(), 
                dynamic_cast< NesterovState * >(layer->getTrainerState()) );
            if(layer->biased()) {
                loadFutureWeights(layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< NesterovState * >(layer->getBiasTrainerState()) );
            }
        }
    }

    // now, we have loaded in weigths + mom * dweights into the weights
    // do forward/backward:
    net->forward(input);
    int numRight = net->calcNumRight(outputData);
    float loss = net->calcLoss(outputData);
    net->backward(outputData);

    // now, calculate the new weights
    for(int layerIdx = numLayers - 2; layerIdx > 0; layerIdx--) {
        Layer *layer = net->getLayer(layerIdx);
        if(!layer->needsBackProp()) {
            break;
        }
        if(layer->needsTrainerState()) {
            updateWeights(layer->getWeightsWrapper(), layer->getGradWeightsWrapper(), 
                dynamic_cast< NesterovState * >(layer->getTrainerState()) );
            if(layer->biased()) {
                updateWeights(layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< NesterovState * >(layer->getBiasTrainerState()) );
            }
        }
    }

    return BatchResult(loss, numRight);
}
VIRTUAL BatchResult Nesterov::trainNet(NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput) {

    ExpectedData expectedData(net, expectedOutput);
    return this->trainNet(net, context, input, &expectedData);
}
VIRTUAL BatchResult Nesterov::trainNetFromLabels(NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels) {

    LabeledData labeledData(net, labels);
    return this->trainNet(net, context, input, &labeledData);
}
VIRTUAL void Nesterov::bindState(NeuralNet *net) {
    NesterovStateMaker stateMaker;
    this->_bindState(net, &stateMaker);
}
STATIC Nesterov *Nesterov::instance(EasyCL *cl, float learningRate) {
    Nesterov *sgd = new Nesterov(cl);
    sgd->setLearningRate(learningRate);
    return sgd;
}
STATIC Nesterov *Nesterov::instance(EasyCL *cl, float learningRate, float momentum) {
    Nesterov *sgd = new Nesterov(cl);
    sgd->setLearningRate(learningRate);
    sgd->setMomentum(momentum);
    return sgd;
}
Nesterov::Nesterov(EasyCL *cl) :
        Trainer(cl),
        momentum(0.0f) {
}

