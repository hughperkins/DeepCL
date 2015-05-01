// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "EasyCL.h"
//#include "ClConvolve2.h"
#include "ActivationFunction.h"
#include "LayerMaker.h"
#include "Timer.h"
#include "StatefulTimer.h"
#include "stringhelper.h"
#include "LayerDimensions.h"
#include "TrainerState.h"

#define VIRTUAL virtual

class TrainerStateMaker;
class Forward;
class Backward;
class BackpropWeights;
class ConvolutionalMaker;
class GpuAdd;
class CopyBuffer;

class ConvolutionalLayer : public Layer {
public:
    EasyCL *const cl; // NOT owned by us
    TrainerState *trainerState; // OWNED by us, we should delete (if non-zero)
    TrainerState *biasTrainerState; // OWNED by us, we should delete (if non-zero)

    Forward *forwardImpl;
    BackpropWeights *backpropWeightsImpl;
    Backward *backwardImpl;

    LayerDimensions dim;
//    ActivationFunction const *const activationFunction;

    float *weights;
    float *bias;
    float *output;
    float *gradInput;
    float *gradWeights;
    float *gradBias;

//    const int filterSize;
//    const int filterSizeSquared;
//    const bool padZeros;

    CLWrapper *weightsWrapper;
    CLWrapper *biasWrapper;
    CLWrapper *outputWrapper;
    CLWrapper *gradInputWrapper;
    CLWrapper *gradWeightsWrapper;
    CLWrapper *gradBiasWrapper;

    int batchSize;
    int allocatedSpaceNumExamples;

    bool weightsCopiedToHost;
    bool biasCopiedToHost;
    bool outputCopiedToHost;
    bool gradInputCopiedToHost;
    bool gradWeightsCopiedToHost;
    bool gradBiasCopiedToHost;

    GpuAdd *gpuAdd;
    CopyBuffer *copyBuffer;

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
    inline int getOutputIndex( int n, int outPlane, int outRow, int outCol ) const {
        return ( ( n
            * dim.numFilters + outPlane )
            * dim.outputImageSize + outRow )
            * dim.outputImageSize + outCol;
    }
    inline float getOutput( int n, int outPlane, int outRow, int outCol ) const {
        return output[ getOutputIndex(n,outPlane, outRow, outCol ) ];
    }

//    ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker );
    // images are organized like [imageId][plane][imagerow][imagecol]
    // filters are organized like [filterid][plane][filterrow][filtercol]
    // output are organized like [imageid][filterid][imagerow][imagecol]
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
    // generated, using cog:
    ConvolutionalLayer( EasyCL *cl, Layer *previousLayer, ConvolutionalMaker *maker );
    VIRTUAL ~ConvolutionalLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float *getGradInput();
    VIRTUAL float *getGradWeights();
    VIRTUAL float *getGradBias();
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL CLWrapper *getGradInputWrapper();
    VIRTUAL CLWrapper *getWeightsWrapper();
    VIRTUAL CLWrapper *getBiasWrapper();
    VIRTUAL CLWrapper *getGradWeightsWrapper();
    VIRTUAL CLWrapper *getGradBiasWrapper();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();
    VIRTUAL bool needsBackProp();
    VIRTUAL float const *getWeights() const;
    VIRTUAL float *getWeights();
    VIRTUAL float *getBias();
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputImageSize() const;
    void randomizeWeights();
    VIRTUAL void print();
    VIRTUAL void printWeights();
    VIRTUAL void printOutput();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void setWeights( float *weights, float *bias );
    VIRTUAL void initWeights( float const*weights );
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getPersistSize() const;
    VIRTUAL void persistToArray(float *array);
    VIRTUAL void unpersistFromArray(float const*array);
    VIRTUAL void initBias( float const*bias );
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasSize() const;
    VIRTUAL float * getOutput();
    VIRTUAL void forward();
    VIRTUAL void backward();
    VIRTUAL void setWeights( CLWrapper *weightWrapper, CLWrapper *biasWrapper );
    VIRTUAL void updateWeights( CLWrapper *weightChangesWrapper, CLWrapper *biasChangesWrapper );
    VIRTUAL std::string asString() const;
    VIRTUAL bool needsTrainerState() const;
    VIRTUAL bool biased();
    VIRTUAL TrainerState *getTrainerState();
    VIRTUAL TrainerState *getBiasTrainerState();
    VIRTUAL void setTrainerState( TrainerStateMaker *trainerStateMaker );

    // [[[end]]]
};

