// Copyright Hugh Perkins 2014 hughperkins at gmail
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

#include "DllImportExport.h"

class OpenCLHelper;
class ConvolutionalMaker;
class LayerMaker;
class RandomTranslatorMaker;
template< typename T> class InputLayerMaker;

#define VIRTUAL virtual
#define STATIC static

class ClConvolve_EXPORT NeuralNet : public Trainable {
public:
    std::vector< Layer *> layers;
    OpenCLHelper *cl;
    int isTraining = true;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    NeuralNet();
    NeuralNet( int numPlanes, int boardSize );
    ~NeuralNet();
    NeuralNet *clone();
    OpenCLHelper *getCl();
    STATIC NeuralNetMould *maker();
    void addLayer( LayerMaker2 *maker );
    FullyConnectedMaker *fullyConnectedMaker();
    PoolingMaker *poolingMaker();
    SquareLossMaker *squareLossMaker();
    CrossEntropyLossMaker *crossEntropyLossMaker();
    SoftMaxMaker *softMaxLossMaker();
    void initWeights( int layerIndex, float *weights, float *biasWeights );
    void initWeights( int layerIndex, float *weights );
    void initBiasWeights( int layerIndex, float *weights );
    void printWeightsAsCode();
    void printBiasWeightsAsCode();
    float calcLoss(float const *expectedValues );
    float calcLossFromLabels(int const *labels );
    EpochMaker *epochMaker();
    VIRTUAL LossLayerMaker *cloneLossLayerMaker( Layer *clonePreviousLayer ) const;
    template< typename T > InputLayer<T> *getFirstLayer();
    Layer *getLastLayer();
    Layer const*getLastLayer() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputBoardSize() const;
    Layer *addLayer( LayerMaker *maker );
    void setBatchSize( int batchSize );
    void setTraining( bool training );
    int calcNumRight( int const *labels );
    void propagate( float const*images);
    void propagate( unsigned char const*images);
    void backPropFromLabels( float learningRate, int const *labels);
    void backProp( float learningRate, float const *expectedResults);
    int getNumLayers();
    float const *getResults( int layer ) const;
    int getInputCubeSize() const;
    int getOutputCubeSize() const;
    float const *getResults() const;
    VIRTUAL int getResultsSize() const;
    void print();
    void printWeights();
    void printOutput();

    // [[[end]]]
};

