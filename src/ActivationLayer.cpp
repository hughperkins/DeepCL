// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NeuralNet.h"
#include "stringhelper.h"

#include "ActivationLayer.h"
#include "ActivationMaker.h"
#include "ActivationPropagate.h"
#include "ActivationBackprop.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

ActivationLayer::ActivationLayer( OpenCLHelper *cl, Layer *previousLayer, ActivationMaker *maker ) :
        Layer( previousLayer, maker ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputImageSize( previousLayer->getOutputImageSize() ),
        outputImageSize( previousLayer->getOutputImageSize() ),
        fn( maker->_activationFunction ),
        cl( cl ),
        results(0),
        errorsForUpstream(0),
        resultsWrapper(0),
        errorsForUpstreamWrapper(0),
        resultsCopiedToHost(false),
        errorsForUpstreamCopiedToHost(false),
        batchSize(0),
        allocatedSize(0) {
    if( inputImageSize == 0 ){
//        maker->net->print();
        throw runtime_error("Error: Activation layer " + toString( layerIndex ) + ": input image size is 0" );
    }
    if( outputImageSize == 0 ){
//        maker->net->print();
        throw runtime_error("Error: Activation layer " + toString( layerIndex ) + ": output image size is 0" );
    }
    activationPropagateImpl = ActivationPropagate::instance( cl, numPlanes, inputImageSize, fn );
    activationBackpropImpl = ActivationBackprop::instance( cl, numPlanes, inputImageSize, fn );
}
VIRTUAL ActivationLayer::~ActivationLayer() {
    delete activationPropagateImpl;
    delete activationBackpropImpl;
    if( resultsWrapper != 0 ) {
        delete resultsWrapper;
    }
    if( results != 0 ) {
        delete[] results;
    }
    if( errorsForUpstreamWrapper != 0 ) {
        delete errorsForUpstreamWrapper;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
}
VIRTUAL std::string ActivationLayer::getClassName() const {
    return "ActivationLayer";
}
VIRTUAL void ActivationLayer::setBatchSize( int batchSize ) {
//    cout << "ActivationLayer::setBatchSize" << endl;
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
    errorsForUpstream = new float[ previousLayer->getResultsSize() ];
    errorsForUpstreamWrapper = cl->wrap( previousLayer->getResultsSize(), errorsForUpstream );
    errorsForUpstreamWrapper->createOnDevice();
}
VIRTUAL int ActivationLayer::getResultsSize() {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL float *ActivationLayer::getResults() {
    if( !resultsCopiedToHost ) {
        resultsWrapper->copyToHost();
        resultsCopiedToHost = true;
    }
    return results;
}
VIRTUAL bool ActivationLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL int ActivationLayer::getResultsSize() const {
//    int outputImageSize = inputImageSize / poolingSize;
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL int ActivationLayer::getOutputImageSize() const {
    return outputImageSize;
}
VIRTUAL int ActivationLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL bool ActivationLayer::providesErrorsForUpstreamWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getErrorsForUpstreamWrapper() {
    return errorsForUpstreamWrapper;
}
VIRTUAL bool ActivationLayer::hasResultsWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getResultsWrapper() {
    return resultsWrapper;
}
VIRTUAL float *ActivationLayer::getErrorsForUpstream() {
    return errorsForUpstream;
}
VIRTUAL ActivationFunction const *ActivationLayer::getActivationFunction() {
    return fn;
}
VIRTUAL void ActivationLayer::propagate() {
    CLWrapper *upstreamResultsWrapper = 0;
    if( previousLayer->hasResultsWrapper() ) {
        upstreamResultsWrapper = previousLayer->getResultsWrapper();
    } else {
        float *upstreamResults = previousLayer->getResults();
        upstreamResultsWrapper = cl->wrap( previousLayer->getResultsSize(), upstreamResults );
        upstreamResultsWrapper->copyToDevice();
    }
    activationPropagateImpl->propagate( batchSize, upstreamResultsWrapper, resultsWrapper );
    if( !previousLayer->hasResultsWrapper() ) {
        delete upstreamResultsWrapper;
    }
}
VIRTUAL void ActivationLayer::backProp( float learningRate ) {
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

    activationBackpropImpl->backpropErrors( batchSize, errorsWrapper, errorsForUpstreamWrapper );

    if( weOwnErrorsWrapper ) {
        delete errorsWrapper;
    }
}
VIRTUAL std::string ActivationLayer::asString() const {
    return "ActivationLayer{ " + fn->getDefineName() + " }";
}
VIRTUAL int ActivationLayer::getPersistSize() const {
    // no weights, so:
    return 0;
}

