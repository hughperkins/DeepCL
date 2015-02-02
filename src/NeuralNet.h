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

#include "DllImportExport.h"

class OpenCLHelper;
class ConvolutionalMaker;
class LayerMaker;
class RandomTranslatorMaker;

#define VIRTUAL virtual
#define STATIC static

class ClConvolve_EXPORT NeuralNet {
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
    OpenCLHelper *getCl();
    STATIC NeuralNetMould *maker();
    template< typename T >InputLayerMaker<T> *inputMaker();
    FullyConnectedMaker *fullyConnectedMaker();
    ConvolutionalMaker *convolutionalMaker();
    PoolingMaker *poolingMaker();
    NormalizationLayerMaker *normalizationMaker();
    RandomPatchesMaker *randomPatchesMaker();
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
    template< typename T > InputLayer<T> *getFirstLayer();
    Layer *getLastLayer();
    Layer *addLayer( LayerMaker *maker );
    void setBatchSize( int batchSize );
    void setTraining( bool training );
    float doEpochFromLabels( float learningRate, int batchSize, int numImages, float const* images, int const *labels );
    float doEpochFromLabels( float learningRate, int batchSize, int numImages, float const* images, int const *labels, int *p_totalCorrect );
    float doEpoch( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults );
    int calcNumRight( int const *labels );
    float doEpochWithCalcTrainingAccuracy( float learningRate, int batchSize, int numImages, float const* images, float const *expectedResults, int const *labels, int *p_totalCorrect );
    template< typename T > void propagate( T const*images);
    void backPropFromLabels( float learningRate, int const *labels);
    void backProp( float learningRate, float const *expectedResults);
    template< typename T > void learnBatch( float learningRate, T const*images, float const *expectedResults );
    template< typename T > void learnBatchFromLabels( float learningRate, T const*images, int const *labels );
    int getNumLayers();
    float const *getResults( int layer ) const;
    int getInputCubeSize() const;
    int getOutputCubeSize() const;
    float const *getResults() const;
    void print();
    void printWeights();
    void printOutput();

    // [[[end]]]
};

