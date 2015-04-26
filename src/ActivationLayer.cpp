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
        output(0),
        errorsForUpstream(0),
        outputWrapper(0),
        errorsForUpstreamWrapper(0),
        outputCopiedToHost(false),
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
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    if( output != 0 ) {
        delete[] output;
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
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    if( output != 0 ) {
        delete[] output;
    }
    if( errorsForUpstreamWrapper != 0 ) {
        delete errorsForUpstreamWrapper;
    }
    if( errorsForUpstream != 0 ) {
        delete[] errorsForUpstream;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    output = new float[ getOutputSize() ];
    outputWrapper = cl->wrap( getOutputSize(), output );
    errorsForUpstream = new float[ previousLayer->getOutputSize() ];
    errorsForUpstreamWrapper = cl->wrap( previousLayer->getOutputSize(), errorsForUpstream );
    errorsForUpstreamWrapper->createOnDevice();
}
VIRTUAL int ActivationLayer::getOutputSize() {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL float *ActivationLayer::getOutput() {
    if( !outputCopiedToHost ) {
        outputWrapper->copyToHost();
        outputCopiedToHost = true;
    }
    return output;
}
VIRTUAL bool ActivationLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL int ActivationLayer::getOutputSize() const {
//    int outputImageSize = inputImageSize / poolingSize;
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL int ActivationLayer::getOutputImageSize() const {
    return outputImageSize;
}
VIRTUAL int ActivationLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL bool ActivationLayer::providesGradInputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getGradInputWrapper() {
    return errorsForUpstreamWrapper;
}
VIRTUAL bool ActivationLayer::hasOutputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getOutputWrapper() {
    return outputWrapper;
}
VIRTUAL float *ActivationLayer::getGradInput() {
    return errorsForUpstream;
}
VIRTUAL ActivationFunction const *ActivationLayer::getActivationFunction() {
    return fn;
}
VIRTUAL void ActivationLayer::propagate() {
    CLWrapper *upstreamOutputWrapper = 0;
    if( previousLayer->hasOutputWrapper() ) {
        upstreamOutputWrapper = previousLayer->getOutputWrapper();
    } else {
        float *upstreamOutput = previousLayer->getOutput();
        upstreamOutputWrapper = cl->wrap( previousLayer->getOutputSize(), upstreamOutput );
        upstreamOutputWrapper->copyToDevice();
    }
    activationPropagateImpl->propagate( batchSize, upstreamOutputWrapper, outputWrapper );
    if( !previousLayer->hasOutputWrapper() ) {
        delete upstreamOutputWrapper;
    }
}
VIRTUAL void ActivationLayer::backProp( float learningRate ) {
    // have no weights to backprop to, just need to backprop the errors

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
        weOwnErrorsWrapper = true;
    }

    activationBackpropImpl->backpropErrors( batchSize, imagesWrapper, errorsWrapper, errorsForUpstreamWrapper );

    if( !previousLayer->hasOutputWrapper() ) {
        delete imagesWrapper;
    }
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

