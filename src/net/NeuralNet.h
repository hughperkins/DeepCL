// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>
#include <algorithm>
#include <iomanip>

#include "net/Trainable.h"

#include "DeepCLDllExport.h"

class EasyCL;
class ConvolutionalMaker;
class LayerMaker;
class RandomTranslatorMaker;
class InputLayerMaker;
class Trainer;
class NeuralNetMould;
class LayerMaker2;
class EpochMaker;
class InputMaker;
class InputLayer;
class OutputData;

#define VIRTUAL virtual
#define STATIC static

/// NeuralNet: main container class for network layers
PUBLICAPI
class DeepCL_EXPORT NeuralNet : public Trainable {
protected:
#ifdef _WIN32
#pragma warning(disable: 4251)
#endif
    std::vector< Layer *> layers;
#ifdef _WIN32
#pragma warning(default: 4251)
#endif
    EasyCL *cl; // NOT owned by us, dont delete
    Trainer *trainer; // NOT owned by us, dont delete

public:
    int isTraining; // = true;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    NeuralNet(EasyCL *cl);
    STATIC NeuralNet *instance(EasyCL *cl);
    STATIC NeuralNet *instance(EasyCL *cl, int numPlanes, int imageSize);
    STATIC NeuralNet *instance3(EasyCL *cl, int numPlanes, int imageSize);
    void deleteMe();
    NeuralNet(EasyCL *cl, int numPlanes, int imageSize);
    ~NeuralNet();
    STATIC NeuralNetMould *maker(EasyCL *cl);
    NeuralNet *clone();
    EasyCL *getCl();
    PUBLICAPI void addLayer(LayerMaker2 *maker);
    PUBLICAPI void initWeights(int layerIndex, float *weights, float *bias);
    PUBLICAPI void initWeights(int layerIndex, float *weights);
    PUBLICAPI void initBias(int layerIndex, float *weights);
    PUBLICAPI float calcLoss(float const *expectedValues);
    PUBLICAPI float calcLossFromLabels(int const *labels);
    float calcLoss(OutputData *outputData);
    int calcNumRight(OutputData *outputData);
    EpochMaker *epochMaker(Trainer *trainer);
    VIRTUAL LossLayerMaker *cloneLossLayerMaker() const;
    PUBLICAPI InputLayer *getFirstLayer();
    PUBLICAPI Layer *getLastLayer();
    PUBLICAPI int getNumLayers() const;
    PUBLICAPI Layer *getLayer(int index);
    PUBLICAPI Layer const*getLastLayer() const;
    PUBLICAPI VIRTUAL int getOutputPlanes() const;
    PUBLICAPI VIRTUAL int getOutputSize() const;
    PUBLICAPI void setBatchSize(int batchSize);
    PUBLICAPI void setTraining(bool training);
    PUBLICAPI int calcNumRight(int const *labels);
    PUBLICAPI void forward(float const*images);
    PUBLICAPI void backwardFromLabels(int const *labels);
    PUBLICAPI void backward(float const *expectedOutput);
    void backward(OutputData *outputData);
    PUBLICAPI int getNumLayers();
    PUBLICAPI float const *getOutput(int layer) const;
    PUBLICAPI int getInputCubeSize() const;
    PUBLICAPI int getOutputCubeSize() const;
    PUBLICAPI float const *getOutput() const;
    PUBLICAPI VIRTUAL int getOutputNumElements() const;
    void print();
    void printWeights();
    void printOutput();
    VIRTUAL void setTrainer(Trainer *trainer);
    void printParamStats();
    PUBLICAPI std::string asString();
    PUBLICAPI const char * asNewCharStar();  // call deepcl_deleteCharStar to delete this

    // [[[end]]]
};

