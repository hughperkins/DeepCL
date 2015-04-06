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

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingLayer::PoolingLayer( OpenCLHelper *cl, Layer *previousLayer, PoolingMaker *maker ) :
        Layer( previousLayer, maker ),
        padZeros( maker->_padZeros ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputImageSize( previousLayer->getOutputImageSize() ),
        poolingSize( maker->_poolingSize ),
        outputImageSize( maker->_padZeros ? ( previousLayer->getOutputImageSize() + maker->_poolingSize - 1 ) / maker->_poolingSize : previousLayer->getOutputImageSize() / maker->_poolingSize ),
        cl( cl ),
        results(0),
        selectors(0),
        errorsForUpstream(0),
        resultsWrapper(0),
        selectorsWrapper(0),
        errorsForUpstreamWrapper(0),
        resultsCopiedToHost(false),
        errorsForUpstreamCopiedToHost(false),
        batchSize(0),
        allocatedSize(0){
    if( inputImageSize == 0 ){
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString( layerIndex ) + ": input image size is 0" );
    }
    if( outputImageSize == 0 ){
//        maker->net->print();
        throw runtime_error("Error: Pooling layer " + toString( layerIndex ) + ": output image size is 0" );
    }
    poolingPropagateImpl = PoolingPropagate::instance( cl, padZeros, numPlanes, inputImageSize, poolingSize );
    poolingBackpropImpl = PoolingBackprop::instance( cl, padZeros, numPlanes, inputImageSize, poolingSize );
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
VIRTUAL std::string PoolingLayer::getClassName() const {
    return "PoolingLayer";
}
VIRTUAL void PoolingLayer::setBatchSize( int batchSize ) {
//    cout << "PoolingLayer::setBatchSize" << endl;
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
    errorsForUpstreamWrapper->createOnDevice();
}
VIRTUAL int PoolingLayer::getResultsSize() {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL float *PoolingLayer::getResults() {
    if( !resultsCopiedToHost ) {
        resultsWrapper->copyToHost();
        resultsCopiedToHost = true;
    }
    return results;
}
VIRTUAL bool PoolingLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL int PoolingLayer::getResultsSize() const {
//    int outputImageSize = inputImageSize / poolingSize;
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL int PoolingLayer::getOutputImageSize() const {
    return outputImageSize;
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
        upstreamResultsWrapper->copyToDevice();
    }
    poolingPropagateImpl->propagate( batchSize, upstreamResultsWrapper, selectorsWrapper, resultsWrapper );
    if( !previousLayer->hasResultsWrapper() ) {
        delete upstreamResultsWrapper;
    }

//    cout << "PoolingLayer::propagate() selectors after propagate: " << endl;
//    for( int i = 0; i < outputImageSize; i++ ) {
//        for( int j = 0; j < outputImageSize; j++ ) {
//            cout << selectors[ i * outputImageSize + j ] << " ";
//        }
//        cout << endl;
//    }

//    cout << "PoolingLayer::propagate() selectorsWrapper after propagate: " << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputImageSize, outputImageSize );
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

//    cout << "PoolingLayer::backProp selectorsWrapper:" << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputImageSize, outputImageSize );

//    int *selectors = reinterpret_cast< int * >( selectorsWrapper->getHostArray() );
//    cout << "PoolingLayer::backProp selectors before copy to host:" << endl;
//    for( int i = 0; i < outputImageSize; i++ ) {
//        for( int j = 0; j < outputImageSize; j++ ) {
//            cout << " " << selectors[i * outputImageSize + j];
//        }
//        cout << endl;
//    }
//    selectorsWrapper->copyToHost();
//    cout << "PoolingLayer::backProp selectors after copy to host:" << endl;
//    for( int i = 0; i < outputImageSize; i++ ) {
//        for( int j = 0; j < outputImageSize; j++ ) {
//            cout << " " << selectors[i * outputImageSize + j];
//        }
//        cout << endl;
//    }
//    selectorsWrapper->copyToDevice();

//    selectorsWrapper->copyToHost();

    poolingBackpropImpl->backpropErrors( batchSize, errorsWrapper, selectorsWrapper, errorsForUpstreamWrapper );

//    errorsForUpstreamWrapper->copyToHost();
//    float *errorsForUpstream = reinterpret_cast< float * >( errorsForUpstreamWrapper->getHostArray() );
//    cout << "errorsForUpstream:" << endl;
//    for( int i = 0; i < inputImageSize; i++ ) {
//        for( int j = 0; j < inputImageSize; j++ ) {
////            cout << " " << errorsForUpstream[i * inputImageSize + j];
//            if( errorsForUpstream[i * inputImageSize + j] != 0 ) {
//                cout << " *";
//            } else {
//                cout << " .";
//            }
//        }
//        cout << endl;
//    }

    if( weOwnErrorsWrapper ) {
        delete errorsWrapper;
    }
}
VIRTUAL std::string PoolingLayer::asString() const {
    return "PoolingLayer{ inputPlanes=" + toString(numPlanes) + " inputImageSize=" + toString(inputImageSize) + " poolingSize=" + toString( poolingSize ) + " }";
}


