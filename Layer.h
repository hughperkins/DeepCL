// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>

#include "MyRandom.h"
#include "ActivationFunction.h"
#include "LayerMaker.h"
#include "stringhelper.h"
#include "OpenCLHelper.h"

class Layer {
public:
//    int batchStart;
//    int batchEnd;

    Layer *previousLayer;
//    Layer *nextLayer;
    const int numPlanes;
    const int boardSize;
    float *results;
    float *weights;
    float *biasWeights;
    const bool biased;
    ActivationFunction const *const activationFunction;
    const int upstreamBoardSize;
    const int upstreamNumPlanes;
    const int layerIndex;
    bool weOwnResults;

    const int boardSizeSquared;
    const int upstreamBoardSizeSquared;

    int batchSize;

//    virtual bool needErrorsBackprop() = 0;

    // results structured like [imageid][outputplane][outputrow][outputcol]
    inline int getResultIndex( int n, int plane, int row, int col ) const {
        return ( ( ( n * numPlanes ) + plane ) * boardSize + row ) * boardSize + col;
    }
    inline float getResult( int n, int plane, int row, int col ) const {
        return results[getResultIndex( n, plane,row,col)];
    }
    inline int getResultsSizePerExample() const {
        return numPlanes * boardSize * boardSize;
    }
    static inline float generateWeight( int fanin ) {
        float rangesize = sqrt(12.0f / (float)fanin) ;
    //        float uniformrand = random() / (float)random.max();     
        float uniformrand = MyRandom::uniform();   
        float result = rangesize * ( uniformrand - 0.5 );
        return result;
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: Layer
    // cppfile: Layer.cpp

    Layer( Layer *previousLayer, LayerMaker const*maker );
    Layer( Layer *previousLayer, ExpectedValuesLayerMaker const*maker );
    virtual ~Layer();
    virtual void setBatchSize( int batchSize );
    virtual bool providesErrorsWrapper() const;
    virtual float *getErrorsForUpstream();
    virtual CLWrapper *getErrorsForUpstreamWrapper();
    virtual bool hasResultsWrapper() const;
    virtual CLWrapper *getResultsWrapper();
    virtual float * getResults();
    virtual int getResultsSize() const;
    int getNumPlanes() const;
    int getBoardSize() const;
    virtual void propagate();
    virtual void print() const;
    virtual void initWeights( float*weights );
    virtual void initBiasWeights( float*biasWeights );
    virtual void printWeightsAsCode() const;
    virtual void printBiasWeightsAsCode() const;
    virtual void printWeights() const;
    virtual void printOutput() const;
    virtual void backPropErrors( float learningRate, Layer *nextLayer );
    virtual int getWeightsSize() const;
    virtual int getBiasWeightsSize() const;
    float calcLoss( float const *expected );

    // [[[end]]]

};

