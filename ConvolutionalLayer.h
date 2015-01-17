// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "OpenCLHelper.h"
//#include "ClConvolve2.h"
#include "ActivationFunction.h"
#include "LayerMaker.h"
#include "Timer.h"
#include "StatefulTimer.h"
#include "stringhelper.h"
#include "LayerDimensions.h"

#define VIRTUAL virtual

class Propagate;
class BackpropErrorsv2;
class BackpropWeights2;

class ConvolutionalLayer : public Layer {
public:
    OpenCLHelper *const cl; // NOT owned by us

    Propagate *propagateimpl;
    BackpropWeights2 *backpropWeightsImpl;
    BackpropErrorsv2 *backpropErrorsImpl;

    LayerDimensions dim;
    ActivationFunction const *const activationFunction;

    float *results;
    float *weights;
    float *biasWeights;

//    const int filterSize;
//    const int filterSizeSquared;
//    const bool padZeros;

    CLWrapper *weightsWrapper;
    CLWrapper *resultsWrapper;
    CLWrapper *errorsForUpstreamWrapper;

    int batchSize;
    int allocatedSpaceNumExamples;

    float *errorsForUpstream;

    bool resultsCopiedToHost;
    bool errorsForUpstreamCopiedToHost;

    inline int getWeightIndex( int filterId, int inputPlane, int filterRow, int filterCol ) const {
        return ( ( filterId 
            * dim.inputPlanes + inputPlane )
            * dim.filterSize + filterRow )
            * dim.filterSize + filterCol;
    }
    inline float getWeight( int filterId, int inputPlane, int filterRow, int filterCol ) const {
//        getWeights();
        return weights[ getWeightIndex( filterId, inputPlane, filterRow, filterCol ) ];
    }
    inline int getResultIndex( int n, int outPlane, int outRow, int outCol ) const {
        return ( ( n
            * dim.numFilters + outPlane )
            * dim.outputBoardSize + outRow )
            * dim.outputBoardSize + outCol;
    }
    inline float getResult( int n, int outPlane, int outRow, int outCol ) const {
        return results[ getResultIndex(n,outPlane, outRow, outCol ) ];
    }

//    ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker );
    // images are organized like [imageId][plane][boardrow][boardcol]
    // filters are organized like [filterid][plane][filterrow][filtercol]
    // results are organized like [imageid][filterid][boardrow][boardcol]
//    inline int getWeightIndex( int outPlane, int inPlane, int filterrow, int filtercol ) const {
//        return ( ( outPlane * upstreamNumPlanes 
//             + inPlane ) * filterSize 
//             + filterrow ) * filterSize
//             + filtercol;
//    }
//    inline float getWeight( int outPlane, int inPlane, int filterrow, int filtercol ) const {
//        return weights[getWeightIndex( outPlane, inPlane, filterrow, filtercol ) ];
//    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: ConvolutionalLayer
    // cppfile: ConvolutionalLayer.cpp

    ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker );
    VIRTUAL ~ConvolutionalLayer();
    VIRTUAL float *getErrorsForUpstream();
    VIRTUAL ActivationFunction const*getActivationFunction();
    VIRTUAL bool providesErrorsForUpstreamWrapper() const;
    VIRTUAL CLWrapper *getErrorsForUpstreamWrapper();
    VIRTUAL float const *getWeights() const;
    VIRTUAL float *getWeights();
    VIRTUAL int getResultsSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputBoardSize() const;
    void randomizeWeights();
    VIRTUAL bool hasResultsWrapper() const;
    VIRTUAL CLWrapper *getResultsWrapper();
    VIRTUAL void print() const;
    VIRTUAL void printWeights() const;
    VIRTUAL void printOutput() const;
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void propagate();
    VIRTUAL float * getResults();
    VIRTUAL void initWeights( float *weights );
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL void initBiasWeights( float *biasWeights );
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasWeightsSize() const;
    VIRTUAL void backProp( float learningRate );

    // [[[end]]]
};

