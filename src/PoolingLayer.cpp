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

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingLayer::PoolingLayer( Layer *previousLayer, PoolingMaker const*maker ) :
        Layer( previousLayer, maker ),
        poolingSize( maker->_poolingSize ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputBoardSize( previousLayer->getOutputBoardSize() ),
        results(0),
        errorsForUpstream(0),
        resultsWrapper(0),
        errorsForUpstreamWrapper(0),
        resultsCopiedToHost(false),
        errorsForUpstreamCopiedToHost(false),
        batchSize(0),
        allocatedSize(0),
        cl( maker->net->getCl() ){
    poolingPropagateImpl = PoolingPropagate::instance( cl, numPlanes, inputBoardSize, poolingSize );
}
VIRTUAL PoolingLayer::~PoolingLayer() {
    delete poolingPropagateImpl;
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
    return numPlanes * inputBoardSize * inputBoardSize / poolingSize / poolingSize;
}
VIRTUAL float *PoolingLayer::getResults() {
    if( !resultsCopiedToHost ) {
        resultsWrapper->copyToHost();
        resultsCopiedToHost = true;
    }
    return results;
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
}
VIRTUAL void PoolingLayer::backProp( float learningRate ) {
    // have no weights to backprop to, just need to backprop the errors
}


