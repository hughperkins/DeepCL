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
class BackpropErrors;
class BackpropWeights;

class ConvolutionalLayer : public Layer {
public:
    OpenCLHelper *const cl; // NOT owned by us

    Propagate *propagateimpl;
    BackpropWeights *backpropWeightsImpl;
    BackpropErrors *backpropErrorsImpl;

    CLKernel *kernelBackPropWeights;
//    CLKernel *kernelBackPropWeights2;
//    CLKernel *kernelBackPropWeights3;
//    CLKernel *kernelBackPropWeights4;
//    CLKernel *kernelBackPropWeightsWithScratch;
    CLKernel *kernelBackpropErrors;
    CLKernel *kernelBackpropBiasWeights;
    CLKernel *kernelAddInPlace;

    LayerDimensions dim;

    const int filterSize;
    const int filterSizeSquared;
    const bool padZeros;

    CLWrapper *weightsWrapper;
    CLWrapper *resultsWrapper;
//    CLWrapper *errorsWrapper;
    CLWrapper *errorsForUpstreamWrapper;

    int allocatedSpaceNumExamples;

//    float *errors;
    float *errorsForUpstream;

    bool resultsCopiedToHost;
//    bool errorsCopiedToHost;
    bool errorsForUpstreamCopiedToHost;
//    bool weightsCopiedToHost;

//    ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker );
    // images are organized like [imageId][plane][boardrow][boardcol]
    // filters are organized like [filterid][plane][filterrow][filtercol]
    // results are organized like [imageid][filterid][boardrow][boardcol]
    inline int getWeightIndex( int outPlane, int inPlane, int filterrow, int filtercol ) const {
        return ( ( outPlane * upstreamNumPlanes 
             + inPlane ) * filterSize 
             + filterrow ) * filterSize
             + filtercol;
    }
    inline float getWeight( int outPlane, int inPlane, int filterrow, int filtercol ) const {
        return weights[getWeightIndex( outPlane, inPlane, filterrow, filtercol ) ];
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: ConvolutionalLayer
    // cppfile: ConvolutionalLayer.cpp

    ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker );
    VIRTUAL ~ConvolutionalLayer();
    VIRTUAL float *getErrorsForUpstream();
    VIRTUAL bool providesErrorsWrapper() const;
    VIRTUAL CLWrapper *getErrorsForUpstreamWrapper();
    VIRTUAL void initWeights( float*weights );
    void randomizeWeights();
    VIRTUAL bool hasResultsWrapper() const;
    VIRTUAL CLWrapper *getResultsWrapper();
    VIRTUAL void print() const;
    VIRTUAL void printWeights() const;
    VIRTUAL void printOutput() const;
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void propagate();
    VIRTUAL float * getResults();
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasWeightsSize() const;
    VIRTUAL void backPropErrors( float learningRate );
    void updateWeightsGpu( CLWrapper* weightChangesWrapper, CLWrapper*weightsWrapper );
    void backPropWeightsCpu( float learningRate, float const *errors, float *weights );
    void backPropWeightsGpu( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, CLWrapper*errorsWrapper, CLWrapper *weightChangesWrapper );
    void doBiasBackpropCpu(float learningRate, float const *results, float const *errors, float *biasWeightChanges );
    void doBiasBackpropGpu(float learningRate, CLWrapper *resultsWrapper, CLWrapper *errorsWrapper, float *biasWeightChanges );

    // [[[end]]]
};

