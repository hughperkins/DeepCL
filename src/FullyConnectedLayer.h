// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "ActivationFunction.h"
#include "StatefulTimer.h"

#define VIRTUAL virtual

class FullyConnectedLayer : public Layer {
public:
    float *results;
    float *weights;
    float *biasWeights;

    float *errorsForUpstream;
    bool allocatedSize;
    ActivationFunction const *const activationFunction;

    // weights like [upstreamPlane][upstreamRow][upstreamCol][outputPlane][outputrow][outputcol]
    inline int getWeightIndex( int prevPlane, int prevRow, int prevCol, int outputPlane, int outputRow, int outputCol ) const {
        int index = ( ( ( ( prevPlane ) * upstreamBoardSize
                       + prevRow ) * upstreamBoardSize
                       + prevCol ) * numPlanes
                       + outputPlane * boardSize 
                       + outputRow ) * boardSize
                       + outputCol;
        return index;
    }
    inline float getWeight( int prevPlane, int prevRow, int prevCol, int outputPlane, int outputRow, int outputCol ) const {
        return weights[getWeightIndex( prevPlane, prevRow, prevCol, outputPlane, outputRow, outputCol )];
    }
    inline int getBiasWeightIndex( int outputPlane, int outputRow, int outputCol ) const {
        int index = (outputPlane * boardSize 
                       + outputRow ) * boardSize
                       + outputCol;
        return index;
    }
    inline float getBiasWeight( int outputPlane, int outputRow, int outputCol ) const {
        return biasWeights[getBiasWeightIndex( outputPlane, outputRow, outputCol ) ];
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: FullyConnectedLayer
    // cppfile: FullyConnectedLayer.cpp

    FullyConnectedLayer( Layer *previousLayer, FullyConnectedMaker const *maker );
    VIRTUAL float *getResults();
    VIRTUAL bool needErrorsBackprop();
    VIRTUAL ActivationFunction const*getActivationFunction();
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasWeightsSize() const;
    VIRTUAL void print() const;
    VIRTUAL void printWeights() const;
    VIRTUAL void printOutput() const;
    void randomizeWeights(int fanIn, float *weights, int numWeights );
    VIRTUAL ~FullyConnectedLayer();
    VIRTUAL float *getErrorsForUpstream();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void propagate();
    VIRTUAL void backProp( float learningRate );

    // [[[end]]]
};

