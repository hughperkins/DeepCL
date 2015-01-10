// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "ConvolutionalLayer.h"
#include "NeuralNet.h"
#include "stringhelper.h"
#include "Propagate.h"
#include "BackpropErrors.h"
#include "BackpropWeights.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

ConvolutionalLayer::ConvolutionalLayer( Layer *previousLayer, ConvolutionalMaker const*maker ) :
        Layer( previousLayer, maker ),
        filterSize( maker->_filterSize ),
        filterSizeSquared( filterSize * filterSize ),
        padZeros( maker->_padZeros ),
        weightsWrapper( 0 ),
        resultsWrapper( 0 ),
//        errorsWrapper( 0 ),
        errorsForUpstreamWrapper( 0 ),
//        errors( 0 ),
        errorsForUpstream( 0 ),
        allocatedSpaceNumExamples( 0 ),
        resultsCopiedToHost( false ),
//        errorsCopiedToHost( false ),
        errorsForUpstreamCopiedToHost( false ),
//        weightsCopiedToHost( false ),
        cl( maker->net->getCl() ) {
        if( padZeros && filterSize % 2 == 0 ) {
            throw std::runtime_error("filter size must be an odd number, if padZeros is true, so either turn off padZeros, or choose a different filtersize :-)");
        }

    dim = LayerDimensions( upstreamNumPlanes, upstreamBoardSize, 
        numPlanes, filterSize, padZeros, biased );
    propagateimpl = Propagate::instance( cl, dim, activationFunction );
    backpropWeightsImpl = BackpropWeights::instance( cl, dim, activationFunction );
    backpropErrorsImpl = BackpropErrors::instance( cl, dim );

    if( filterSize > upstreamBoardSize ) {
            throw std::runtime_error("filter size cannot be larger than upstream board size: " + toString( filterSize) +
                " > " + toString(upstreamBoardSize) );
    }
    biasWeights = new float[ getBiasWeightsSize() ];
    weights = new float[ getWeightsSize() ];
//    std::cout << " convolutional layer " << layerIndex << " allocating weights size " << getWeightsSize() << std::endl;
    randomizeWeights();
    weightsWrapper = cl->wrap( getWeightsSize(), weights );
    weightsWrapper->copyToDevice();
}

VIRTUAL ConvolutionalLayer::~ConvolutionalLayer() {
    if( weightsWrapper != 0 ) {
        delete weightsWrapper;
    }
    if( resultsWrapper != 0 ) {
        delete resultsWrapper;
    }
    if( errorsForUpstreamWrapper != 0 ) {
        delete errorsForUpstreamWrapper;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
    delete propagateimpl;
    delete backpropWeightsImpl;
    delete backpropErrorsImpl;
}
VIRTUAL float *ConvolutionalLayer::getErrorsForUpstream() {
    if( !errorsForUpstreamCopiedToHost ) {
        std::cout << "copying errorsForUpstream to host, from GPU" << std::endl;
        errorsForUpstreamWrapper->copyToHost();
        errorsForUpstreamCopiedToHost = true;
    }
    return errorsForUpstream;
}
VIRTUAL bool ConvolutionalLayer::providesErrorsWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getErrorsForUpstreamWrapper() {
    return errorsForUpstreamWrapper;
}
VIRTUAL void ConvolutionalLayer::initWeights( float*weights ) {
    Layer::initWeights( weights );
    weightsWrapper->copyToDevice();
}

// filters are organized like [filterid][plane][row][col]
void ConvolutionalLayer::randomizeWeights() {
//        std::cout << "convolutional layer randomzing weights" << std::endl;
    int fanin = upstreamNumPlanes * filterSize * filterSize;
    const int numThisLayerWeights = getWeightsSize();
    for( int i = 0; i < numThisLayerWeights; i++ ) {
        weights[i] = generateWeight( fanin );
    }
    for( int i = 0; i < numPlanes; i++ ) {
        biasWeights[i] = generateWeight( fanin );
    }
}
VIRTUAL bool ConvolutionalLayer::hasResultsWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getResultsWrapper() {
    return resultsWrapper;
}
VIRTUAL void ConvolutionalLayer::print() const {
    std::cout << "ConvolutionalLayer numFilters " << numPlanes << " filtersize " << filterSize << 
        " padZeros " << padZeros << " biased " << biased << " outputBoardSize " << boardSize << std::endl;
    printWeights();
    if( results != 0 ) {
        printOutput();
    }
}
VIRTUAL void ConvolutionalLayer::printWeights() const {
    std::cout << "  weights: " << std::endl;
// filters are organized like [filterid][plane][row][col]
    for( int filter = 0; filter < std::min( 5, numPlanes ); filter++ ) {
       std::cout << "    filter " << filter << std::endl;
       if( biased ) {
           std::cout << "       bias=" << biasWeights[filter] << std::endl;            
       }
       for( int plane = 0; plane < std::min(5,upstreamNumPlanes); plane++ ) {
           if( upstreamNumPlanes > 1 ) std::cout << "    inplane " << plane << std::endl;
            for( int i = 0; i < std::min(5,filterSize); i++ ) {
                std::cout << "      ";
                for( int j = 0; j < std::min(5,filterSize); j++ ) {
                   std::cout << getWeight( filter, plane, i, j ) << " ";
                }
                if( filterSize > 5 ) {
                   std::cout << " ...";
                }
                std::cout << std::endl;
            }
            if( filterSize > 5 ) {
               std::cout << " ..." << std::endl;
            }
        }
        if( upstreamNumPlanes > 5 ) std::cout << " ... other inplanes ... " << std::endl;
    }
    if( numPlanes > 5 ) std::cout << " ... other filters ... " << std::endl;
 }
VIRTUAL void ConvolutionalLayer::printOutput() const { 
    if( results == 0 ) {
        return;
    }
    std::cout << "  outputs: " << std::endl;
// results are organized like [imageid][filterid][row][col]
    for( int n = 0; n < std::min( 5, batchSize ); n++ ) {
        std::cout << "    n: " << n << std::endl;
        for( int plane = 0; plane < std::min(5,numPlanes); plane++ ) {
            if( numPlanes > 1 ) std::cout << "      plane " << plane << std::endl;
            if( boardSize == 1 ) {
                 std::cout << "        " << getResult(n, plane, 0, 0 ) << std::endl;
            } else {
                for( int i = 0; i < std::min(5,boardSize); i++ ) {
                    std::cout << "      ";
                    for( int j = 0; j < std::min(5,boardSize); j++ ) {
                        std::cout << getResult( n, plane, i, j ) << " ";
                    }
                    if( boardSize > 5 ) std::cout << " ... ";
                    std::cout << std::endl;
                }
                if( boardSize > 5 ) std::cout << " ... " << std::endl;
            }
            if( numPlanes > 5 ) std::cout << " ... other planes ... " << std::endl;
        }
        if( batchSize > 5 ) std::cout << " ... other n ... " << std::endl;
    }
}
VIRTUAL void ConvolutionalLayer::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSpaceNumExamples ) {
        this->batchSize = batchSize;
        return;
    }
    if( results != 0 ) {
        delete[] results;
    }
    if( resultsWrapper != 0 ) {
        delete resultsWrapper;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
    if( errorsForUpstreamWrapper != 0 ) {
        delete errorsForUpstreamWrapper;
    }
    this->batchSize = batchSize;
    results = new float[getResultsSize()];
    resultsWrapper = cl->wrap( getResultsSize(), results );
//        std::cout << " layer " << layerIndex << " allocating results size " << getResultsSize() << std::endl;
    weOwnResults = true;
    if( layerIndex > 1 ) {
        errorsForUpstream = new float[ previousLayer->getResultsSize() ];
        errorsForUpstreamWrapper = cl->wrap( previousLayer->getResultsSize(), errorsForUpstream );
    }
    this->allocatedSpaceNumExamples = batchSize;
}
VIRTUAL void ConvolutionalLayer::propagate() {
//    if( boardSizeSquared <= cl->getMaxWorkgroupSize() ) {
////        propagate2();
//    } else {
//  //      propagate1();
//    }
//    propagate1();
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ", START");

    CLWrapper *upstreamWrapper = 0;
    if( previousLayer->hasResultsWrapper() ) {
//            std::cout << "layer " << previousLayer->layerIndex << " has resultsWrapper" << std::endl;
        upstreamWrapper = previousLayer->getResultsWrapper();
    } else {
//            std::cout << "layer " << previousLayer->layerIndex << " has no resultsWrapper" << std::endl;
        upstreamWrapper = cl->wrap( previousLayer->getResultsSize(), (float *)previousLayer->getResults() );
        upstreamWrapper->copyToDevice();
    }
    CLFloatWrapper *biasWeightsWrapper = 0;
    if( biased ) {
        biasWeightsWrapper = cl->wrap( getBiasWeightsSize(), biasWeights );
        biasWeightsWrapper->copyToDevice();
    }
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ", copied to device");
    propagateimpl->propagate( batchSize, upstreamWrapper, weightsWrapper, biasWeightsWrapper, resultsWrapper );
    StatefulTimer::instance()->timeCheck("    propagate layer " + toString( layerIndex ) + ",  after clFinish");

    if( !previousLayer->hasResultsWrapper() ) {
        delete upstreamWrapper;
    }
    if( biased ) {
        delete biasWeightsWrapper;
    }
    resultsCopiedToHost = false;
}
VIRTUAL float * ConvolutionalLayer::getResults() {
    if( !resultsCopiedToHost ) {
//            std::cout << "layer " << layerIndex << " copying results to host " << std::endl;
        resultsWrapper->copyToHost();
        resultsCopiedToHost = true;
    }
    return results;
};
VIRTUAL int ConvolutionalLayer::getWeightsSize() const {
    return numPlanes * upstreamNumPlanes * filterSize * filterSize;
}
VIRTUAL int ConvolutionalLayer::getBiasWeightsSize() const {
    return numPlanes;
}

// weights:     [outPlane][upstreamPlane][filterRow][filterCol]
//       aggregate over:  [outRow][outCol][n]
// biasweights: [outPlane]
//       aggregate over:  [upstreamPlane][filterRow][filterCol][outRow][outCol][n]

VIRTUAL void ConvolutionalLayer::backPropErrors( float learningRate ) {
//        Timer timer;
    StatefulTimer::instance()->timeCheck("backproperrors(): start backprop, layer " + toString( layerIndex ) );

    CLWrapper *biasWeightsWrapper = 0;
    if( dim.biased ) {
        biasWeightsWrapper = cl->wrap( getBiasWeightsSize(), biasWeights );
        biasWeightsWrapper->copyToDevice();
    }

    CLWrapper *imagesWrapper = 0;
    if( previousLayer->hasResultsWrapper() ) {
        imagesWrapper = previousLayer->getResultsWrapper();
    } else {
        imagesWrapper = cl->wrap( previousLayer->getResultsSize(), previousLayer->getResults() );
        imagesWrapper->copyToDevice();
    }

    CLWrapper *errorsWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if( nextLayer->providesErrorsWrapper() ) {
        errorsWrapper = nextLayer->getErrorsForUpstreamWrapper();
    } else {
        errorsWrapper = cl->wrap( getResultsSize(), nextLayer->getErrorsForUpstream() );
        errorsWrapper->copyToDevice();
        weOwnErrorsWrapper = true;
    }

    if( layerIndex > 1 ) {
        backpropErrorsImpl->backpropErrors( batchSize, weightsWrapper, biasWeightsWrapper, errorsWrapper, errorsForUpstreamWrapper );
        StatefulTimer::instance()->timeCheck("backproperrors(): calced errors for upstream, layer " + toString( layerIndex ) );
    }

    backpropWeightsImpl->backpropWeights( batchSize, learningRate, errorsWrapper, resultsWrapper, imagesWrapper,   weightsWrapper, biasWeightsWrapper );
    StatefulTimer::instance()->timeCheck("backproperrors(): done weight backprop, layer " + toString( layerIndex ) );

    if( dim.biased ) {
        biasWeightsWrapper->copyToHost();
        delete biasWeightsWrapper;
    }
    if( !previousLayer->hasResultsWrapper() ) {
        delete imagesWrapper;
    }
    if( weOwnErrorsWrapper ) {
        delete errorsWrapper;
    }
    StatefulTimer::instance()->timeCheck("backproperrors(): updated weights, layer " + toString( layerIndex ) );
}

void ConvolutionalLayer::backPropWeightsCpu( float learningRate, float const *errors, float *weights ) {
//        Timer timer;
    const float learningMultiplier = learningRate / batchSize / sqrt( boardSize * boardSize );
//        const bool debug = false;
    const int halfFilterSize = filterSize >> 1;
    const int margin = padZeros ? halfFilterSize : 0;
    StatefulTimer::instance()->timeCheck(" backpropweightscpu start, layer " + toString( layerIndex ) );
    for( int outPlane = 0; outPlane < numPlanes; outPlane++ ) {
        for( int upstreamPlane = 0; upstreamPlane < upstreamNumPlanes; upstreamPlane++ ) {
            for( int filterRow = 0; filterRow < filterSize; filterRow++ ) {
                for( int filterCol = 0; filterCol < filterSize; filterCol++ ) {
                    int weightIndex = getWeightIndex( outPlane, upstreamPlane, filterRow, filterCol );
//                        if( filterRow != 1 || filterCol > 1 ) {
//                            weights[weightIndex] = 0;
//                            continue;
//                        }
                    float thiswchange = 0;
                    // weights:     [outPlane][upstreamPlane][filterRow][filterCol]
                    //       aggregate over:  [outRow][outCol][n]
                    for( int outRow = 0; outRow < boardSize; outRow++ ) {
                        int upstreamRow = outRow - margin + filterRow;
                        for( int outCol = 0; outCol < boardSize; outCol++ ) {
                            int upstreamCol = outCol - margin + filterCol;
                            for( int n = 0; n < batchSize; n++ ) {
                                int resultIndex = getResultIndex( n, outPlane, outRow, outCol );
                                float error = errors[resultIndex];
                                float actualOutput = results[resultIndex];
                                float activationDerivative = activationFunction->calcDerivative( actualOutput );
//                                    float activationDerivative = 1 - actualOutput * actualOutput;
                                float upstreamResult = previousLayer->getResult( n, upstreamPlane, upstreamRow, upstreamCol );
                                float thisimagethiswchange = upstreamResult * activationDerivative *
                                error;
                                thiswchange += thisimagethiswchange;
//    if(debug)std::cout << "outPlane=" << outPlane << " inPlane=" << upstreamPlane << " filterpos=" << filterRow << "," << filterCol
//       << " outpos=" << outRow << "," << outCol << " n=" << n << " resindex " << resultIndex << " error=" << error
//       << " actualoutput=" << actualOutput << " upstreampos=" << upstreamRow <<"," << upstreamCol << " upstreamResult=" << upstreamResult << " thisimagethiswchange="
//       << thisimagethiswchange << std::endl;
                            }
                        }
                    }
//                        weights[ weightIndex ] -= learningRate * thiswchange / batchSize / sqrt( boardSize * boardSize );
                    weights[ weightIndex ] += - thiswchange * learningMultiplier;
                }
            }
        }
    }
//        timer.timeCheck("did backprop to ourselves v2");
    StatefulTimer::instance()->timeCheck(" backpropweightscpu end, layer " + toString( layerIndex ) );
}

