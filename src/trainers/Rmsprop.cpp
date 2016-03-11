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
#include "trainers/RmspropStateMaker.h"
#include "trainers/RmspropState.h"
#include "trainers/Rmsprop.h"
#include "loss/IAcceptsLabels.h"
#include "batch/NetAction.h"
#include "clmath/CLMathWrapper.h"
#include "batch/BatchData.h"

//#include "test/Sampler.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


VIRTUAL Rmsprop::~Rmsprop() {
}
VIRTUAL std::string Rmsprop::asString() {
    return "Rmsprop{ learningRate=" + toString(learningRate) + " }";
}
VIRTUAL void Rmsprop::updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
        RmspropState *trainerState) {

    int numWeights = trainerState->numWeights;
    float *working = new float[ numWeights ];
    CLWrapper *workingWrapper = cl->wrap(numWeights, working);
    workingWrapper->createOnDevice();

    CLMathWrapper clWeights(weightsWrapper);
    CLMathWrapper clGradWeights(gradWeightsWrapper);
    CLMathWrapper clMeanSquares(trainerState->meanSquareWrapper);
    CLMathWrapper clWorking(workingWrapper);

    // following all happens on gpu, via clmathwrapper:
    clWorking = clGradWeights;
    clWorking.squared();
    clWorking *= 0.1f; // I guess this should be a hyper-parameter?
    clMeanSquares *= 0.9f;
    clMeanSquares += clWorking;

    clWorking = clMeanSquares;
    clWorking.sqrt();
    clWorking.inv();
    clWorking *= clGradWeights;
    clWorking *= - learningRate;
    clWeights += clWorking;

    delete workingWrapper;
    delete[] working;
}
VIRTUAL BatchResult Rmsprop::trainNet(NeuralNet *net, TrainingContext *context,
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
                dynamic_cast< RmspropState * >(layer->getTrainerState()) );
            if(layer->biased()) {
                updateWeights(layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< RmspropState * >(layer->getBiasTrainerState()) );
            }
        }
    }
    return BatchResult(loss, numRight);
}
VIRTUAL BatchResult Rmsprop::trainNet(NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput) {
    ExpectedData expectedData(net, expectedOutput);
    return this->trainNet(net, context, input, &expectedData);
}
VIRTUAL BatchResult Rmsprop::trainNetFromLabels(NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels) {
    LabeledData labeledData(net, labels);
    return this->trainNet(net, context, input, &labeledData);
}
VIRTUAL void Rmsprop::bindState(NeuralNet *net) {
    RmspropStateMaker stateMaker;
    this->_bindState(net, &stateMaker);
}
STATIC Rmsprop *Rmsprop::instance(EasyCL *cl, float learningRate) {
    Rmsprop *sgd = new Rmsprop(cl);
    sgd->setLearningRate(learningRate);
    return sgd;
}
Rmsprop::Rmsprop(EasyCL *cl) :
        Trainer(cl) {
}

