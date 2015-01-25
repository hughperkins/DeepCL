// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "NeuralNet.h"
#include "Layer.h"
#include "PoolingLayer.h"
#include "PoolingPropagate.h"
#include "PoolingBackprop.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingLayer::PoolingLayer( Layer *previousLayer, PoolingMaker const*maker ) :
        Layer( previousLayer, maker ),
        padZeros( maker->_padZeros ),
        poolingSize( maker->_poolingSize ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputBoardSize( previousLayer->getOutputBoardSize() ),
        outputBoardSize( maker->_padZeros ? ( previousLayer->getOutputBoardSize() + maker->_poolingSize - 1 ) / maker->_poolingSize : previousLayer->getOutputBoardSize() / maker->_poolingSize ),
        results(0),
        errorsForUpstream(0),
        selectors(0),
        resultsWrapper(0),
        selectorsWrapper(0),
        errorsForUpstreamWrapper(0),
        resultsCopiedToHost(false),
        errorsForUpstreamCopiedToHost(false),
        batchSize(0),
        allocatedSize(0),
        cl( maker->net->getCl() ){
    poolingPropagateImpl = PoolingPropagate::instance( cl, padZeros, numPlanes, inputBoardSize, poolingSize );
    poolingBackpropImpl = PoolingBackprop::instance( cl, padZeros, numPlanes, inputBoardSize, poolingSize );
}
VIRTUAL PoolingLayer::~PoolingLayer() {
    delete poolingPropagateImpl;
    delete poolingBackpropImpl;
    if( resultsWrapper != 0 ) {
        delete resultsWrapper;
    }
    if( results != 0 ) {
        delete[] results;
    }
    if( selectorsWrapper != 0 ) {
        delete selectorsWrapper;
    }
    if( selectors != 0 ) {
        delete[] selectors;
    }
    if( errorsForUpstreamWrapper != 0 ) {
        delete errorsForUpstreamWrapper;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
}
VIRTUAL void PoolingLayer::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSize ) {
        this->batchSize = batchSize;
        return;
    }
        if( resultsWrapper != 0 ) {
        delete resultsWrapper;
    }
    if( results != 0 ) {
        delete[] results;
    }
    if( selectorsWrapper != 0 ) {
        delete selectorsWrapper;
    }
    if( selectors != 0 ) {
        delete[] selectors;
    }
    if( errorsForUpstreamWrapper != 0 ) {
        delete errorsForUpstreamWrapper;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    results = new float[ getResultsSize() ];
    resultsWrapper = cl->wrap( getResultsSize(), results );
    selectors = new int[ getResultsSize() ];
    selectorsWrapper = cl->wrap( getResultsSize(), selectors );
    errorsForUpstream = new float[ previousLayer->getResultsSize() ];
    errorsForUpstreamWrapper = cl->wrap( previousLayer->getResultsSize(), errorsForUpstream );
}
VIRTUAL int PoolingLayer::getResultsSize() {
    return batchSize * numPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL float *PoolingLayer::getResults() {
    if( !resultsCopiedToHost ) {
        resultsWrapper->copyToHost();
        resultsCopiedToHost = true;
    }
    return results;
}
VIRTUAL int PoolingLayer::getResultsSize() const {
    int outputBoardSize = inputBoardSize / poolingSize;
    return batchSize * numPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL int PoolingLayer::getOutputBoardSize() const {
    return outputBoardSize;
}
VIRTUAL int PoolingLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int PoolingLayer::getPersistSize() const {
    return 0;
}
VIRTUAL bool PoolingLayer::providesErrorsForUpstreamWrapper() const {
    return true;
}
VIRTUAL CLWrapper *PoolingLayer::getErrorsForUpstreamWrapper() {
    return errorsForUpstreamWrapper;
}
VIRTUAL bool PoolingLayer::hasResultsWrapper() const {
    return true;
}
VIRTUAL CLWrapper *PoolingLayer::getResultsWrapper() {
    return resultsWrapper;
}
VIRTUAL float *PoolingLayer::getErrorsForUpstream() {
    return errorsForUpstream;
}
VIRTUAL ActivationFunction const *PoolingLayer::getActivationFunction() {
    return previousLayer->getActivationFunction(); // I guess???
}
VIRTUAL void PoolingLayer::propagate() {
    CLWrapper *upstreamResultsWrapper = 0;
    if( previousLayer->hasResultsWrapper() ) {
        upstreamResultsWrapper = previousLayer->getResultsWrapper();
    } else {
        float *upstreamResults = previousLayer->getResults();
        upstreamResultsWrapper = cl->wrap( previousLayer->getResultsSize(), upstreamResults );
    }
    poolingPropagateImpl->propagate( batchSize, upstreamResultsWrapper, selectorsWrapper, resultsWrapper );
    if( !previousLayer->hasResultsWrapper() ) {
        delete upstreamResultsWrapper;
    }
}
VIRTUAL void PoolingLayer::backProp( float learningRate ) {
    // have no weights to backprop to, just need to backprop the errors

    CLWrapper *errorsWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if( nextLayer->providesErrorsForUpstreamWrapper() ) {
        errorsWrapper = nextLayer->getErrorsForUpstreamWrapper();
    } else {
        errorsWrapper = cl->wrap( getResultsSize(), nextLayer->getErrorsForUpstream() );
        errorsWrapper->copyToDevice();
        weOwnErrorsWrapper = true;
    }

    poolingBackpropImpl->backpropErrors( batchSize, errorsWrapper, selectorsWrapper, errorsForUpstreamWrapper );

    if( weOwnErrorsWrapper ) {
        delete errorsWrapper;
    }
}
VIRTUAL std::string PoolingLayer::asString() const {
    return "PoolingLayer{ inputPlanes=" + toString(numPlanes) + " inputBoardSize=" + toString(inputBoardSize) + " poolingSize=" + toString( poolingSize ) + " }";
}


