// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>
#include <algorithm>

#include "Layer.h"
#include "NeuralNetMould.h"
#include "EpochMaker.h"
#include "ConvolutionalLayer.h"
#include "InputLayer.h"
#include "Trainable.h"
#include "InputLayerMaker.h"
#include "ConvolutionalMaker.h"
#include "RandomTranslationsMaker.h"
#include "RandomPatchesMaker.h"
#include "NormalizationLayerMaker.h"
#include "FullyConnectedMaker.h"
#include "PoolingMaker.h"
#include "LayerMaker.h"
#include "ActivationMaker.h"
#include "DropoutMaker.h"

#include "DeepCLDllExport.h"

class OpenCLHelper;
class ConvolutionalMaker;
class LayerMaker;
class RandomTranslatorMaker;
class InputLayerMaker;

#define VIRTUAL virtual
#define STATIC static

/// NeuralNet: main container class for network layers
PUBLICAPI
class DeepCL_EXPORT NeuralNet : public Trainable {
protected:
#ifdef _WIN32
#pragma warning( disable: 4251 )
#endif
    std::vector< Layer *> layers;
#ifdef _WIN32
#pragma warning( default: 4251 )
#endif
    OpenCLHelper *cl;
public:
    int isTraining; // = true;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PUBLICAPI NeuralNet();
    NeuralNet( int gpu );
    PUBLICAPI NeuralNet( int numPlanes, int imageSize );
    ~NeuralNet();
    NeuralNet *clone();
    OpenCLHelper *getCl();
    STATIC NeuralNetMould *maker();
    PUBLICAPI void addLayer( LayerMaker2 *maker );
    PUBLICAPI void initWeights( int layerIndex, float *weights, float *biasWeights );
    PUBLICAPI void initWeights( int layerIndex, float *weights );
    PUBLICAPI void initBiasWeights( int layerIndex, float *weights );
    void printWeightsAsCode();
    void printBiasWeightsAsCode();
    PUBLICAPI float calcLoss(float const *expectedValues );
    PUBLICAPI float calcLossFromLabels(int const *labels );
    EpochMaker *epochMaker();
    VIRTUAL LossLayerMaker *cloneLossLayerMaker() const;
    PUBLICAPI InputLayer *getFirstLayer();
    PUBLICAPI Layer *getLastLayer();
    PUBLICAPI int getNumLayers() const;
    PUBLICAPI Layer *getLayer( int index );
    PUBLICAPI Layer const*getLastLayer() const;
    PUBLICAPI VIRTUAL int getOutputPlanes() const;
    PUBLICAPI VIRTUAL int getOutputImageSize() const;
    PUBLICAPI void setBatchSize( int batchSize );
    PUBLICAPI void setTraining( bool training );
    PUBLICAPI int calcNumRight( int const *labels );
    PUBLICAPI void forward( float const*images);
    PUBLICAPI void backwardFromLabels( float learningRate, int const *labels);
    PUBLICAPI void backward( float learningRate, float const *expectedOutput);
    PUBLICAPI int getNumLayers();
    PUBLICAPI float const *getOutput( int layer ) const;
    PUBLICAPI int getInputCubeSize() const;
    PUBLICAPI int getOutputCubeSize() const;
    PUBLICAPI float const *getOutput() const;
    PUBLICAPI VIRTUAL int getOutputSize() const;
    void print();
    void printWeights();
    void printOutput();
    PUBLICAPI std::string asString();

    // [[[end]]]
};

