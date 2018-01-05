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
#include "trainers/AdadeltaStateMaker.h"
#include "trainers/AdadeltaState.h"
#include "trainers/Adadelta.h"
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


VIRTUAL Adadelta::~Adadelta() {
}
VIRTUAL std::string Adadelta::asString() {
    return "Adadelta{ learningRate=" + toString(learningRate) + " }";
}
VIRTUAL void Adadelta::updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
        AdadeltaState *trainerState) {
    // need to calculate
    // sumGradSquared = decay * sumGradSquared + (1 - decay) * grad.square()
    // update = - sumUpdateSquared.sqrt() / sumGradSquared.sqrt() * grad
    // sumUpdateSquared = decay * sumUpdateSquared + (1 - decay) * update.squared()
    // weights += update

    int numWeights = trainerState->numWeights;
    float *working = new float[ numWeights ];
    CLWrapper *workingWrapper = cl->wrap(numWeights, working);
    workingWrapper->createOnDevice();

    CLMathWrapper clWeights(weightsWrapper);
    CLMathWrapper clGradWeights(gradWeightsWrapper);
    CLMathWrapper clSumGradSquared(trainerState->sumGradSquaredWrapper);
    CLMathWrapper clSumUpdateSquared(trainerState->sumUpdateSquaredWrapper);
    CLMathWrapper clWorking(workingWrapper);

    // following all happens on gpu, via clmathwrapper:
    clWorking = clGradWeights;
    clWorking.squared();
    clWorking *= (1 - decay);
    clSumGradSquared *= decay;
    clSumGradSquared += clWorking;

    clWorking = clSumGradSquared;
    clWorking += 0.0000001f;
    clWorking.inv();
    clWorking *= clSumUpdateSquared;
    clWorking.sqrt();
    clWorking *= clGradWeights;
    clWorking *= - 1;

    clWeights += clWorking;

    clSumUpdateSquared *= decay;
    clWorking.squared();
    clWorking *= (1 - decay);
    clSumUpdateSquared += clWorking;

    delete workingWrapper;
    delete[] working;
}
VIRTUAL BatchResult Adadelta::trainNet(NeuralNet *net, TrainingContext *context,
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
                dynamic_cast< AdadeltaState * >(layer->getTrainerState()) );
            if(layer->biased()) {
                updateWeights(layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< AdadeltaState * >(layer->getBiasTrainerState()) );
            }
        }
    }
    return BatchResult(loss, numRight);
}
VIRTUAL BatchResult Adadelta::trainNet(NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput) {
    ExpectedData expectedData(net, expectedOutput);
    return this->trainNet(net, context, input, &expectedData);
}
VIRTUAL BatchResult Adadelta::trainNetFromLabels(NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels) {
    LabeledData labeledData(net, labels);
    return this->trainNet(net, context, input, &labeledData);
}
VIRTUAL void Adadelta::bindState(NeuralNet *net) {
    AdadeltaStateMaker stateMaker;
    this->_bindState(net, &stateMaker);
}
STATIC Adadelta *Adadelta::instance(EasyCL *cl, float decay) {
    Adadelta *trainer = new Adadelta(cl, decay);
    return trainer;
}
Adadelta::Adadelta(EasyCL *cl, float decay) :
        Trainer(cl),
        decay(decay) {
    this->setLearningRate(0.0f);
}

