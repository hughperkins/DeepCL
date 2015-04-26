// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ConvolutionalLayer.h"
#include "NeuralNet.h"
#include "stringhelper.h"
#include "Propagate.h"
#include "WeightsHelper.h"
#include "BackpropErrorsv2.h"
#include "BackpropWeights2.h"
#include "SGD.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

ConvolutionalLayer::ConvolutionalLayer( OpenCLHelper *cl, Layer *previousLayer, ConvolutionalMaker *maker ) :
        Layer( previousLayer, maker ),
//        filterSize( maker->_filterSize ),
//        filterSizeSquared( filterSize * filterSize ),
//        padZeros( maker->_padZeros ),
        cl( cl ),
        weightsTrainer( 0 ),
        biasWeightsTrainer( 0 ),
        backpropErrorsImpl(0),
        activationFunction( maker->_activationFunction ),
        output(0),
        weights(0),
        biasWeights(0),
        weightsWrapper( 0 ),
        outputWrapper( 0 ),
        gradInputWrapper( 0 ),
        batchSize( 0 ),
        allocatedSpaceNumExamples( 0 ),
        gradInput( 0 ),
        outputCopiedToHost( false ),
        gradInputCopiedToHost( false ),
        weightsCopiedToHost(false) {
    dim.setInputPlanes( previousLayer->getOutputPlanes() )
        .setInputImageSize( previousLayer->getOutputImageSize() )
        .setNumFilters( maker->_numFilters )
        .setFilterSize( maker->_filterSize )
        .setBiased( maker->_biased )
        .setPadZeros( maker->_padZeros );
    if( dim.padZeros && dim.filterSize % 2 == 0 ) {
        throw std::runtime_error("filter size must be an odd number, if padZeros is true, so either turn off padZeros, or choose a different filtersize :-)");
    }
    weightsTrainer = new SGD( cl, getWeightsSize() ); // so it doesnt crash...
    biasWeightsTrainer = new SGD( cl, getBiasWeightsSize() );

//    dim = LayerDimensions( upstreamNumPlanes, upstreamImageSize, 
//        numPlanes, filterSize, padZeros, biased );
    propagateimpl = Propagate::instance( cl, dim, activationFunction );
    backpropWeightsImpl = BackpropWeights2::instance( cl, dim );
    if( previousLayer->needsBackProp() ) {
        backpropErrorsImpl = BackpropErrorsv2::instance( cl, dim, previousLayer->getActivationFunction() );
    }

    if( dim.filterSize > dim.inputImageSize ) {
            throw std::runtime_error("filter size cannot be larger than upstream image size: " + toString( dim.filterSize) +
                " > " + toString(dim.inputImageSize) );
    }
    biasWeights = new float[ getBiasWeightsSize() ];
    weights = new float[ getWeightsSize() ];
    randomizeWeights();
    weightsWrapper = cl->wrap( getWeightsSize(), weights );
    weightsWrapper->copyToDevice();
    weightsCopiedToHost = true;
}
VIRTUAL ConvolutionalLayer::~ConvolutionalLayer() {
    if( weightsWrapper != 0 ) {
        delete weightsWrapper;
    }
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    if( output != 0 ) {
        delete[] output;
    }
    if( weights != 0 ) {
        delete[] weights;
    }
    if( biasWeights != 0 ) {
        delete[] biasWeights;
    }
    if( gradInputWrapper != 0 ) {
        delete gradInputWrapper;
    }
    if( gradInput != 0 ) {
        delete[] gradInput;
    }
    delete propagateimpl;
    delete backpropWeightsImpl;
    delete backpropErrorsImpl;
    delete weightsTrainer;
    delete biasWeightsTrainer;
}
VIRTUAL std::string ConvolutionalLayer::getClassName() const {
    return "ConvolutionalLayer";
}
VIRTUAL ActivationFunction const*ConvolutionalLayer::getActivationFunction() {
    return activationFunction;
}
VIRTUAL float *ConvolutionalLayer::getGradInput() {
    if( !gradInputCopiedToHost ) {
        std::cout << "copying gradInput to host, from GPU" << std::endl;
        gradInputWrapper->copyToHost();
        gradInputCopiedToHost = true;
    }
    return gradInput;
}
VIRTUAL bool ConvolutionalLayer::providesgradInputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getGradInputWrapper() {
    return gradInputWrapper;
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
VIRTUAL float *ConvolutionalLayer::getBiasWeights() {
    //if( !biasWeightsCopiedToHost ) {
//        cout << "copying weights to host" << endl;
      //  cl->finish();
       // biasWeightsWrapper->copyToHost();
    //}
    return biasWeights;
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
    for( int i = 0; i < dim.numFilters; i++ ) {
        biasWeights[i] = WeightsHelper::generateWeight( fanin );
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
           std::cout << "       bias=" << biasWeights[filter] << std::endl;            
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
VIRTUAL void ConvolutionalLayer::printOutput() const { 
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
                 std::cout << "        " << getResult(n, plane, 0, 0 ) << std::endl;
            } else {
                for( int i = 0; i < std::min(5, dim.outputImageSize); i++ ) {
                    std::cout << "      ";
                    for( int j = 0; j < std::min(5, dim.outputImageSize); j++ ) {
                        std::cout << getResult( n, plane, i, j ) << " ";
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
    if( output != 0 ) {
        delete[] output;
    }
    output = new float[getOutputSize()];
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    outputWrapper = cl->wrap( getOutputSize(), output );
    if( gradInput != 0 ) {
        delete[] gradInput;
    }
    if( gradInputWrapper != 0 ) {
        delete gradInputWrapper;
    }
    if( layerIndex > 1 ) {
        gradInput = new float[ previousLayer->getOutputSize() ];
        gradInputWrapper = cl->wrap( previousLayer->getOutputSize(), gradInput );
    }
}
VIRTUAL void ConvolutionalLayer::propagate() {
    if( batchSize == 0 ) {
        throw runtime_error("Need to call setBatchSize(size) before calling propagate etc");
    }
//    if( imageSizeSquared <= cl->getMaxWorkgroupSize() ) {
////        propagate2();
//    } else {
//  //      propagate1();
//    }
//    propagate1();
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ", START");

    CLWrapper *upstreamWrapper = 0;
    if( previousLayer->hasOutputWrapper() ) {
//            std::cout << "layer " << previousLayer->layerIndex << " has outputWrapper" << std::endl;
        upstreamWrapper = previousLayer->getOutputWrapper();
    } else {
//            std::cout << "layer " << previousLayer->layerIndex << " has no outputWrapper" << std::endl;
        upstreamWrapper = cl->wrap( previousLayer->getOutputSize(), (float *)previousLayer->getOutput() );
        upstreamWrapper->copyToDevice();
    }
    CLFloatWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        biasWeightsWrapper = cl->wrap( getBiasWeightsSize(), biasWeights );
        biasWeightsWrapper->copyToDevice();
    }
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ", copied to device");
    propagateimpl->propagate( batchSize, upstreamWrapper, weightsWrapper, biasWeightsWrapper, outputWrapper );
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ",  after clFinish");

    if( !previousLayer->hasOutputWrapper() ) {
        delete upstreamWrapper;
    }
    if( dim.biased ) {
        delete biasWeightsWrapper;
    }
    outputCopiedToHost = false;
}
VIRTUAL float * ConvolutionalLayer::getOutput() {
    if( !outputCopiedToHost ) {
//            std::cout << "layer " << layerIndex << " copying output to host " << std::endl;
        outputWrapper->copyToHost();
        outputCopiedToHost = true;
    }
    return output;
};
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
        return getWeightsSize() + getBiasWeightsSize();
    } else {
        return getWeightsSize();
    }
}
VIRTUAL void ConvolutionalLayer::persistToArray(float *array) {
    float const*weights = getWeights();
//    float const*biasWeights = getBiasWeights();
    memcpy( array, weights, sizeof(float) * getWeightsSize() );
    if( dim.biased ) {
        memcpy( array + getWeightsSize(), biasWeights, sizeof(float) * getBiasWeightsSize() );
    }
}
VIRTUAL void ConvolutionalLayer::unpersistFromArray(float const*array) {
    float const*newweights = array;
    initWeights( newweights );
    if( dim.biased ) {
        float const*newbiasWeights = array + getWeightsSize();
        initBiasWeights( newbiasWeights );
    }
}
VIRTUAL void ConvolutionalLayer::initBiasWeights( float const*biasWeights ) {
    int biasWeightsSize = dim.numFilters;
    memcpy( this->biasWeights, biasWeights, sizeof(float) * biasWeightsSize );
//    biasWeightsWrapper->copyToDevice();
}
VIRTUAL int ConvolutionalLayer::getWeightsSize() const {
    return dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;
}
VIRTUAL int ConvolutionalLayer::getBiasWeightsSize() const {
    if( dim.biased ) {
        return dim.numFilters;
    } else {
        return 0;
    }
}

// weights:     [outPlane][upstreamPlane][filterRow][filterCol]
//       aggregate over:  [outRow][outCol][n]
// biasweights: [outPlane]
//       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]

VIRTUAL void ConvolutionalLayer::backProp( float learningRate ) {
//        Timer timer;
    StatefulTimer::instance()->timeCheck("backprop(): start, layer " + toString( layerIndex ) );

    CLWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        biasWeightsWrapper = cl->wrap( getBiasWeightsSize(), biasWeights );
        biasWeightsWrapper->copyToDevice();
    }

    CLWrapper *imagesWrapper = 0;
    if( previousLayer->hasOutputWrapper() ) {
        imagesWrapper = previousLayer->getOutputWrapper();
    } else {
        imagesWrapper = cl->wrap( previousLayer->getOutputSize(), previousLayer->getOutput() );
        imagesWrapper->copyToDevice();
    }

    CLWrapper *errorsWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if( nextLayer->providesGradInputWrapper() ) {
        errorsWrapper = nextLayer->getGradInputWrapper();
    } else {
        errorsWrapper = cl->wrap( getOutputSize(), nextLayer->getGradInput() );
        errorsWrapper->copyToDevice();
//        int outputSize = getOutputSize();
//        for( int i = 0; i < outputSize; i++ ) {
//            cout << "convolutional::backproperrors errorsfromupstream[" << i << "]=" << nextLayer->getGradInput()[i] << endl;
//        }
        weOwnErrorsWrapper = true;
    }
    if( previousLayer->needsBackProp() ) {
        backpropErrorsImpl->backpropErrors( batchSize, imagesWrapper, errorsWrapper, weightsWrapper, gradInputWrapper );
        StatefulTimer::instance()->timeCheck("backproperrors(): calced errors for upstream, layer " + ::toString( layerIndex ) );
    }

    backpropWeightsImpl->backpropWeights( batchSize, learningRate, errorsWrapper, imagesWrapper,  weightsWrapper, biasWeightsWrapper );
    weightsCopiedToHost = false;
    StatefulTimer::instance()->timeCheck("backproperrors(): done weight backprop, layer " + ::toString( layerIndex ) );

    if( dim.biased ) {
        biasWeightsWrapper->copyToHost();
        delete biasWeightsWrapper;
    }
    if( !previousLayer->hasOutputWrapper() ) {
        delete imagesWrapper;
    }
    if( weOwnErrorsWrapper ) {
        delete errorsWrapper;
    }
    StatefulTimer::instance()->timeCheck("backproperrors(): updated weights, layer " + ::toString( layerIndex ) );
}

VIRTUAL std::string ConvolutionalLayer::asString() const {
    return "ConvolutionalLayer{ " + toString( dim ) + " " + activationFunction->getDefineName() + " }";
}

VIRTUAL bool ConvolutionalLayer::needsTrainer() const {
    return true;
}

VIRTUAL void ConvolutionalLayer::setTrainer( Trainer *weightsTrainer, Trainer *biasWeightsTrainer ) {
    delete weightsTrainer;
    delete biasWeightsTrainer;
    this->weightsTrainer = weightsTrainer;
    this->biasWeightsTrainer = biasWeightsTrainer;
}

ostream &operator<<( ostream &os, ConvolutionalLayer &layer ) {
    os << "ConvolutionalLayer { " << layer.dim << " }";
    return os;
}

