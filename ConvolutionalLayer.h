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

class ConvolutionalLayer : public Layer {
public:
    OpenCLHelper *const cl; // NOT owned by us
    CLKernel *kernelConvolve;
    CLKernel *kernelBackPropWeights;
//    CLKernel *kernelBackPropWeights2;
//    CLKernel *kernelBackPropWeights3;
//    CLKernel *kernelBackPropWeights4;
    CLKernel *kernelBackPropWeightsWithScratch;
    CLKernel *kernelBackPropWeightsWithScratchAndBias;
    CLKernel *kernelBackpropErrors;
    CLKernel *kernelBackpropBiasWeights;
    CLKernel *kernelAddInPlace;

    const int filterSize;
    const int filterSizeSquared;
    const bool padZeros;

    CLWrapper *weightsWrapper;
    CLWrapper *resultsWrapper;

    int allocatedSpaceNumExamples;

    bool resultsCopiedToHost;
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
    // cppfile: /home/user/git/ClConvolve/ConvolutionalLayer.cpp

    ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker );
    virtual ~ConvolutionalLayer();
    virtual void initWeights( float*weights );
    void randomizeWeights();
    virtual bool hasResultsWrapper() const;
    virtual CLWrapper *getResultsWrapper();
    virtual void print() const;
    virtual void printWeights() const;
    virtual void printOutput() const;
    virtual void setBatchSize( int batchSize );
    virtual void propagate();
    virtual float * getResults();
    virtual int getWeightsSize() const;
    virtual int getBiasWeightsSize() const;
    virtual void calcErrors( float const *expected, float *errors );
    virtual void backPropErrors( float learningRate, float const *errors, float *errorsForUpstream );
    void updateWeightsGpu( CLWrapper* weightChangesWrapper, CLWrapper*weightsWrapper );
    void backPropWeightsCpu( float learningRate, float const *errors, float *weights );
    void backPropWeightsGpu( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, float const*errors, CLWrapper *weightChangesWrapper );
    void backPropWeightsGpuWithScratch( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, float const*errors, CLWrapper *weightChangesWrapper );
    void backPropWeightsGpuWithScratchAndBias( float learningRate, CLWrapper *imagesWrapper, CLWrapper *resultsWrapper, float const*errors, CLWrapper *weightChangesWrapper, float *biasWeightChanges );
    virtual bool needErrorsBackprop();
    void calcErrorsForUpstreamGpu( CLWrapper *weightsWrapper, float const *const errors, float *const errorsForUpstream );
    void calcErrorsForUpstreamCpu( float const *const weights, float const *const errors, float *errorsForUpstream );
    void doBiasBackpropCpu(float learningRate, float const *results, float const *errors, float *biasWeightChanges );
    void doBiasBackpropGpu(float learningRate, CLWrapper *resultsWrapper, float const *errors, float *biasWeightChanges );

    // [[[end]]]
};

