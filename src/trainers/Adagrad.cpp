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
#include "trainers/AdagradStateMaker.h"
#include "trainers/AdagradState.h"
#include "trainers/Adagrad.h"
#include "loss/IAcceptsLabels.h"
#include "batch/NetAction.h"
#include "clmath/CLMathWrapper.h"
#include "batch/BatchData.h"

//#include "test/Sampler.h"

using namespace std;
using namespace easycl;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


VIRTUAL Adagrad::~Adagrad() {
}
VIRTUAL void Adagrad::setFudgeFactor(float fudgeFactor) {
    this->fudgeFactor = fudgeFactor;
}
VIRTUAL std::string Adagrad::asString() {
    return "Adagrad{ learningRate=" + toString(learningRate) + ", fudgeFactor=" + 
        toString(fudgeFactor) + " }"; // if you have a better name, let me know :-)
}
VIRTUAL void Adagrad::updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
        AdagradState *trainerState) {

    int numWeights = trainerState->numWeights;
    float *working = new float[ numWeights ];
    CLWrapper *workingWrapper = cl->wrap(numWeights, working);
    workingWrapper->createOnDevice();

    CLMathWrapper clWeights(weightsWrapper);
    CLMathWrapper clGradWeights(gradWeightsWrapper);
    CLMathWrapper clSumSquares(trainerState->sumSquaresWrapper);
    CLMathWrapper clWorking(workingWrapper);

    // following all happens on gpu, via clmathwrapper:
    clWorking = clGradWeights;
    clWorking.squared();
    clSumSquares += clWorking;

    clWorking = clSumSquares;
    clWorking.sqrt();
    clWorking.inv();
    clWorking *= clGradWeights;
    clWorking *= - learningRate;
    clWeights += clWorking;

    delete workingWrapper;
    delete[] working;
}
VIRTUAL BatchResult Adagrad::trainNet(NeuralNet *net, TrainingContext *context,
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
                dynamic_cast< AdagradState * >(layer->getTrainerState()) );
            if(layer->biased()) {
                updateWeights(layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< AdagradState * >(layer->getBiasTrainerState()) );
            }
        }
    }
    return BatchResult(loss, numRight);
}
VIRTUAL BatchResult Adagrad::trainNet(NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput) {
    ExpectedData expectedData(net, expectedOutput);
    return this->trainNet(net, context, input, &expectedData);
}
VIRTUAL BatchResult Adagrad::trainNetFromLabels(NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels) {
    LabeledData labeledData(net, labels);
    return this->trainNet(net, context, input, &labeledData);
}
VIRTUAL void Adagrad::bindState(NeuralNet *net) {
    AdagradStateMaker stateMaker(fudgeFactor);
    this->_bindState(net, &stateMaker);
}
STATIC Adagrad *Adagrad::instance(EasyCL *cl, float learningRate) {
    Adagrad *sgd = new Adagrad(cl);
    sgd->setLearningRate(learningRate);
    return sgd;
}
Adagrad::Adagrad(EasyCL *cl) :
        Trainer(cl),
        fudgeFactor(0.000001f) {
}

