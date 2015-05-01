// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ConvolutionalLayer.h"
#include "ConvolutionalMaker.h"
#include "NeuralNet.h"
#include "stringhelper.h"
#include "Forward.h"
#include "WeightsHelper.h"
#include "Backward.h"
#include "BackpropWeights.h"
#include "TrainerStateMaker.h"
#include "TrainerState.h"
#include "SGDState.h"
#include "GpuAdd.h"
#include "CopyBuffer.h"
#include "Layer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

ConvolutionalLayer::ConvolutionalLayer( EasyCL *cl, Layer *previousLayer, ConvolutionalMaker *maker ) :
        Layer( previousLayer, maker ),
//        filterSize( maker->_filterSize ),
//        filterSizeSquared( filterSize * filterSize ),
//        padZeros( maker->_padZeros ),
        cl( cl ),
        trainerState( 0 ),
        biasTrainerState( 0 ),
        forwardImpl(0),
        backwardImpl(0),
//        activationFunction( maker->_activationFunction ),

        weights(0),
        bias(0),
        output(0),
        gradInput( 0 ),
        gradWeights( 0 ),
        gradBias( 0 ),

        weightsWrapper( 0 ),
        biasWrapper( 0 ),
        outputWrapper( 0 ),
        gradInputWrapper( 0 ),
        gradWeightsWrapper( 0 ),
        gradBiasWrapper( 0 ),

        batchSize( 0 ),
        allocatedSpaceNumExamples( 0 ),

        weightsCopiedToHost( false ),
        biasCopiedToHost( false ),
        outputCopiedToHost( false ),
        gradInputCopiedToHost( false ),
        gradWeightsCopiedToHost( false ),
        gradBiasCopiedToHost( false ) 
            {
    dim.setInputPlanes( previousLayer->getOutputPlanes() )
        .setInputImageSize( previousLayer->getOutputImageSize() )
        .setNumFilters( maker->_numFilters )
        .setFilterSize( maker->_filterSize )
        .setBiased( maker->_biased )
        .setPadZeros( maker->_padZeros );
    if( dim.padZeros && dim.filterSize % 2 == 0 ) {
        throw std::runtime_error("filter size must be an odd number, if padZeros is true, so either turn off padZeros, or choose a different filtersize :-)");
    }
//    weightsTrainer = new SGD( cl, getWeightsSize() ); // so it doesnt crash...
//    biasTrainer = new SGD( cl, getBiasSize() );

//    dim = LayerDimensions( upstreamNumPlanes, upstreamImageSize, 
//        numPlanes, filterSize, padZeros, biased );
    forwardImpl = Forward::instance( cl, dim );
    backpropWeightsImpl = BackpropWeights::instance( cl, dim );
    if( previousLayer->needsBackProp() ) {
        backwardImpl = Backward::instance( cl, dim );
    }

    if( dim.filterSize > dim.inputImageSize ) {
            throw std::runtime_error("filter size cannot be larger than upstream image size: " + toString( dim.filterSize) +
                " > " + toString(dim.inputImageSize) );
    }
    weights = new float[ getWeightsSize() ];
    if( dim.biased ) {
        bias = new float[ getBiasSize() ];
    }
    randomizeWeights();

    weightsWrapper = cl->wrap( getWeightsSize(), weights );
    weightsWrapper->copyToDevice();
    weightsCopiedToHost = true;

    if( dim.biased ) {
        biasWrapper = cl->wrap( getBiasSize(), bias );
        biasWrapper->copyToDevice();
        biasCopiedToHost = true;
    }

    gradWeights = new float[ getWeightsSize() ];
    gradWeightsWrapper = cl->wrap( getWeightsSize(), gradWeights );
    gradWeightsWrapper->createOnDevice();

    if( dim.biased ) {
        gradBias = new float[ getBiasSize() ];
        gradBiasWrapper = cl->wrap( getBiasSize(), gradBias );
        gradBiasWrapper->createOnDevice();
    }

    gpuAdd = new GpuAdd( cl );
    copyBuffer = new CopyBuffer( cl );
}
VIRTUAL ConvolutionalLayer::~ConvolutionalLayer() {
    delete gpuAdd;
    delete copyBuffer;

    delete weightsWrapper;
    delete biasWrapper;
    delete outputWrapper;
    delete gradInputWrapper;
    delete gradWeightsWrapper;
    delete gradBiasWrapper;

    delete[] output;
    delete[] weights;
    delete[] bias;
    delete[] gradInput;
    delete[] gradWeights;
    delete[] gradBias;

    delete forwardImpl;
    delete backpropWeightsImpl;
    delete backwardImpl;
    delete trainerState;
    delete biasTrainerState;
}
VIRTUAL std::string ConvolutionalLayer::getClassName() const {
    return "ConvolutionalLayer";
}
//VIRTUAL ActivationFunction const*ConvolutionalLayer::getActivationFunction() {
//    return activationFunction;
//}
VIRTUAL float *ConvolutionalLayer::getGradInput() {
    if( !gradInputCopiedToHost ) {
        std::cout << "copying gradInput to host, from GPU" << std::endl;
        gradInputWrapper->copyToHost();
        gradInputCopiedToHost = true;
    }
    return gradInput;
}
VIRTUAL float *ConvolutionalLayer::getGradWeights() {
    if( !gradWeightsCopiedToHost ) {
        std::cout << "copying gradWeights to host, from GPU" << std::endl;
        gradWeightsWrapper->copyToHost();
        gradWeightsCopiedToHost = true;
    }
    return gradWeights;
}
VIRTUAL float *ConvolutionalLayer::getGradBias() {
    if( !gradBiasCopiedToHost ) {
        std::cout << "copying gradBias to host, from GPU" << std::endl;
        gradBiasWrapper->copyToHost();
        gradBiasCopiedToHost = true;
    }
    return gradBias;
}
VIRTUAL bool ConvolutionalLayer::providesGradInputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getGradInputWrapper() {
    return gradInputWrapper;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getWeightsWrapper() {
    return weightsWrapper;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getBiasWrapper() {
    return biasWrapper;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getGradWeightsWrapper() {
    return gradWeightsWrapper;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getGradBiasWrapper() {
    return gradBiasWrapper;
}
VIRTUAL bool ConvolutionalLayer::hasOutputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getOutputWrapper() {
    return outputWrapper;
}
VIRTUAL bool ConvolutionalLayer::needsBackProp() {
    return true;
}
VIRTUAL float const *ConvolutionalLayer::getWeights() const {
    if( !weightsCopiedToHost ) {
        throw std::runtime_error("weights not copied to host, and htis is const object, so cannot copy");
    }
    return weights;
}
VIRTUAL float *ConvolutionalLayer::getWeights() {
    if( !weightsCopiedToHost ) {
//        cout << "copying weights to host" << endl;
        cl->finish();
        weightsWrapper->copyToHost();
    }
    return weights;
}
VIRTUAL float *ConvolutionalLayer::getBias() {
    //if( !biasCopiedToHost ) {
//        cout << "copying weights to host" << endl;
      //  cl->finish();
       // biasWrapper->copyToHost();
    //}
    return bias;
}
VIRTUAL int ConvolutionalLayer::getOutputSize() const {
    return batchSize * dim.outputCubeSize;
}
VIRTUAL int ConvolutionalLayer::getOutputPlanes() const {
    return dim.numFilters;
}
VIRTUAL int ConvolutionalLayer::getOutputImageSize() const {
    return dim.outputImageSize;
}
// filters are organized like [filterid][plane][row][col]
void ConvolutionalLayer::randomizeWeights() {
//        std::cout << "convolutional layer randomzing weights" << std::endl;
    int fanin = dim.inputPlanes * dim.filterSize * dim.filterSize;
    const int numThisLayerWeights = getWeightsSize();
    for( int i = 0; i < numThisLayerWeights; i++ ) {
        weights[i] = WeightsHelper::generateWeight( fanin );
    }
    if( dim.biased ) {
        for( int i = 0; i < dim.numFilters; i++ ) {
            bias[i] = WeightsHelper::generateWeight( fanin );
        }
    }
}
VIRTUAL void ConvolutionalLayer::print() {
    std::cout << "ConvolutionalLayer " << dim << std::endl;
    printWeights();
    if( output != 0 ) {
        printOutput();
    }
}
VIRTUAL void ConvolutionalLayer::printWeights() {
    std::cout << "  weights: " << std::endl;
    getWeights();
// filters are organized like [filterid][plane][row][col]
    for( int filter = 0; filter < std::min( 5, dim.numFilters ); filter++ ) {
       std::cout << "    filter " << filter << std::endl;
       if( dim.biased ) {
           std::cout << "       bias=" << bias[filter] << std::endl;            
       }
       for( int plane = 0; plane < std::min(5, dim.inputPlanes); plane++ ) {
           if( dim.inputPlanes > 1 ) std::cout << "    inplane " << plane << std::endl;
            for( int i = 0; i < std::min(5, dim.filterSize); i++ ) {
                std::cout << "      ";
                for( int j = 0; j < std::min(5, dim.filterSize); j++ ) {
                   std::cout << getWeight( filter, plane, i, j ) << " ";
                }
                if( dim.filterSize > 5 ) {
                   std::cout << " ...";
                }
                std::cout << std::endl;
            }
            if( dim.filterSize > 5 ) {
               std::cout << " ..." << std::endl;
            }
        }
        if( dim.inputPlanes > 5 ) std::cout << " ... other inplanes ... " << std::endl;
    }
    if( dim.numFilters > 5 ) std::cout << " ... other filters ... " << std::endl;
 }
VIRTUAL void ConvolutionalLayer::printOutput() { 
    if( output == 0 ) {
        return;
    }
    //    getOutput();
    std::cout << "  outputs: " << std::endl;
// output are organized like [imageid][filterid][row][col]
    for( int n = 0; n < std::min( 5, batchSize ); n++ ) {
        std::cout << "    n: " << n << std::endl;
        for( int plane = 0; plane < std::min(5, dim.numFilters ); plane++ ) {
            if( dim.numFilters > 1 ) std::cout << "      plane " << plane << std::endl;
            if( dim.outputImageSize == 1 ) {
                 std::cout << "        " << getOutput(n, plane, 0, 0 ) << std::endl;
            } else {
                for( int i = 0; i < std::min(5, dim.outputImageSize); i++ ) {
                    std::cout << "      ";
                    for( int j = 0; j < std::min(5, dim.outputImageSize); j++ ) {
                        std::cout << getOutput( n, plane, i, j ) << " ";
                    }
                    if( dim.outputImageSize > 5 ) std::cout << " ... ";
                    std::cout << std::endl;
                }
                if( dim.outputImageSize > 5 ) std::cout << " ... " << std::endl;
            }
            if( dim.numFilters > 5 ) std::cout << " ... other planes ... " << std::endl;
        }
        if( batchSize > 5 ) std::cout << " ... other n ... " << std::endl;
    }
}
VIRTUAL void ConvolutionalLayer::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSpaceNumExamples ) {
        this->batchSize = batchSize;
        return;
    }

    this->batchSize = batchSize;
    this->allocatedSpaceNumExamples = batchSize;

    delete outputWrapper;
    delete[] output;

    delete gradInputWrapper;
    delete[] gradInput;

    output = new float[getOutputSize()];
    outputWrapper = cl->wrap( getOutputSize(), output );

    if( layerIndex > 1 ) {
        gradInput = new float[ previousLayer->getOutputSize() ];
        gradInputWrapper = cl->wrap( previousLayer->getOutputSize(), gradInput );
    }
}
VIRTUAL void ConvolutionalLayer::setWeights( float *weights, float *bias ) {
    initWeights( weights );
    if( dim.biased ) {
        initBias( bias );
    }
}
VIRTUAL void ConvolutionalLayer::initWeights( float const*weights ) {
    int weightsSize = dim.filtersSize;
    memcpy( this->weights, weights, sizeof(float) * weightsSize );
    weightsWrapper->copyToDevice();
}
VIRTUAL int ConvolutionalLayer::getOutputCubeSize() const {
    return dim.outputCubeSize;
}
VIRTUAL int ConvolutionalLayer::getPersistSize() const {
    if( dim.biased ) {
        return getWeightsSize() + getBiasSize();
    } else {
        return getWeightsSize();
    }
}
VIRTUAL void ConvolutionalLayer::persistToArray(float *array) {
    float const*weights = getWeights();
//    float const*bias = getBias();
    memcpy( array, weights, sizeof(float) * getWeightsSize() );
    if( dim.biased ) {
        memcpy( array + getWeightsSize(), bias, sizeof(float) * getBiasSize() );
    }
}
VIRTUAL void ConvolutionalLayer::unpersistFromArray(float const*array) {
    float const*newweights = array;
    initWeights( newweights );
    if( dim.biased ) {
        float const*newbias = array + getWeightsSize();
        initBias( newbias );
    }
}
VIRTUAL void ConvolutionalLayer::initBias( float const*bias ) {
    int biasSize = dim.numFilters;
    memcpy( this->bias, bias, sizeof(float) * biasSize );
    biasWrapper->copyToDevice();
}
VIRTUAL int ConvolutionalLayer::getWeightsSize() const {
    return dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;
}
VIRTUAL int ConvolutionalLayer::getBiasSize() const {
    if( dim.biased ) {
        return dim.numFilters;
    } else {
        return 0;
    }
}
VIRTUAL float * ConvolutionalLayer::getOutput() {
    if( !outputCopiedToHost ) {
        outputWrapper->copyToHost();
        outputCopiedToHost = true;
    }
    return output;
};
VIRTUAL void ConvolutionalLayer::forward() {
    if( batchSize == 0 ) {
        throw runtime_error("Need to call setBatchSize(size) before calling forward etc");
    }
    StatefulTimer::instance()->timeCheck("    forward layer " + toString( layerIndex ) + ", START");

    CLWrapper *upstreamWrapper = 0;
    if( previousLayer->hasOutputWrapper() ) {
//            std::cout << "layer " << previousLayer->layerIndex << " has outputWrapper" << std::endl;
        upstreamWrapper = previousLayer->getOutputWrapper();
    } else {
//            std::cout << "layer " << previousLayer->layerIndex << " has no outputWrapper" << std::endl;
        upstreamWrapper = cl->wrap( previousLayer->getOutputSize(), (float *)previousLayer->getOutput() );
        upstreamWrapper->copyToDevice();
    }
    StatefulTimer::instance()->timeCheck("    forward layer " + toString( layerIndex ) + ", copied to device");
    forwardImpl->forward( batchSize, upstreamWrapper, weightsWrapper, biasWrapper, outputWrapper );
    StatefulTimer::instance()->timeCheck("    forward layer " + toString( layerIndex ) + ",  after clFinish");

    if( !previousLayer->hasOutputWrapper() ) {
        delete upstreamWrapper;
    }
    outputCopiedToHost = false;
}
VIRTUAL void ConvolutionalLayer::backward() {
    StatefulTimer::instance()->timeCheck("backprop(): start, layer " + toString( layerIndex ) );

    CLWrapper *inputWrapper = 0;
    if( previousLayer->hasOutputWrapper() ) {
        inputWrapper = previousLayer->getOutputWrapper();
    } else {
        inputWrapper = cl->wrap( previousLayer->getOutputSize(), previousLayer->getOutput() );
        inputWrapper->copyToDevice();
    }

    CLWrapper *gradOutputWrapper = 0;
    bool weOwnGradOutputWrapper = false;
    if( nextLayer->providesGradInputWrapper() ) {
        gradOutputWrapper = nextLayer->getGradInputWrapper();
    } else {
        gradOutputWrapper = cl->wrap( getOutputSize(), nextLayer->getGradInput() );
        gradOutputWrapper->copyToDevice();
        weOwnGradOutputWrapper = true;
    }

    if( previousLayer->needsBackProp() ) {
        backwardImpl->backward( batchSize, inputWrapper, gradOutputWrapper, weightsWrapper, gradInputWrapper );
        StatefulTimer::instance()->timeCheck("backproperrors(): calced gradInput, layer " + ::toString( layerIndex ) );
    }

    backpropWeightsImpl->calcGradWeights( batchSize, gradOutputWrapper, inputWrapper,  gradWeightsWrapper, gradBiasWrapper );
    StatefulTimer::instance()->timeCheck("backproperrors(): done calc gradWeights, layer " + ::toString( layerIndex ) );

    gradWeightsCopiedToHost = false;
    gradBiasCopiedToHost = false;

    if( !previousLayer->hasOutputWrapper() ) {
        delete inputWrapper;
    }
    if( weOwnGradOutputWrapper ) {
        delete gradOutputWrapper;
    }
}
VIRTUAL void ConvolutionalLayer::setWeights( CLWrapper *weightWrapper, CLWrapper *biasWrapper ) {
    copyBuffer->copy( getWeightsSize(), weightWrapper, this->weightsWrapper );
    if( dim.biased ) {
        copyBuffer->copy( getBiasSize(), biasWrapper, this->biasWrapper );
    }
    weightsCopiedToHost = false;
    biasCopiedToHost = false;
    StatefulTimer::instance()->timeCheck("ConvolutionalLayer::setWeights(): set weights, layer " + ::toString( layerIndex ) );
}
VIRTUAL void ConvolutionalLayer::updateWeights( CLWrapper *weightChangesWrapper, CLWrapper *biasChangesWrapper ) {
    gpuAdd->add( getWeightsSize(), weightsWrapper, weightChangesWrapper );
    if( dim.biased ) {
        gpuAdd->add( getBiasSize(), biasWrapper, biasChangesWrapper );
    }
    weightsCopiedToHost = false;
    biasCopiedToHost = false;
    StatefulTimer::instance()->timeCheck("ConvolutionalLayer::updateWeights(): updated weights, layer " + ::toString( layerIndex ) );
}
VIRTUAL std::string ConvolutionalLayer::asString() const {
    return "ConvolutionalLayer{ " + toString( dim ) + " }";
}
VIRTUAL bool ConvolutionalLayer::needsTrainerState() const {
    return true;
}
VIRTUAL bool ConvolutionalLayer::biased() {
    return dim.biased;
}
VIRTUAL TrainerState *ConvolutionalLayer::getTrainerState() {
    return trainerState;
}
VIRTUAL TrainerState *ConvolutionalLayer::getBiasTrainerState() {
    return biasTrainerState;
}
VIRTUAL void ConvolutionalLayer::setTrainerState( TrainerStateMaker *trainerStateMaker ) {
    delete trainerState;
    delete biasTrainerState;
    this->trainerState = trainerStateMaker->instance( cl, getWeightsSize() );
    if( dim.biased ) {
        this->biasTrainerState = trainerStateMaker->instance( cl, getBiasSize() );
    }
}

