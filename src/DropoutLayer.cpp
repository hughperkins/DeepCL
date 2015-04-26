// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "NeuralNet.h"
#include "Layer.h"
#include "DropoutLayer.h"
#include "DropoutPropagate.h"
#include "DropoutBackprop.h"
#include "RandomSingleton.h"
#include "MultiplyBuffer.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

DropoutLayer::DropoutLayer( OpenCLHelper *cl, Layer *previousLayer, DropoutMaker *maker ) :
        Layer( previousLayer, maker ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputImageSize( previousLayer->getOutputImageSize() ),
        dropRatio( maker->_dropRatio ),
        outputImageSize( previousLayer->getOutputImageSize() ),
        random( RandomSingleton::instance() ),
        cl( cl ),
        masks(0),
        output(0),
        errorsForUpstream(0),
        maskWrapper(0),
        outputWrapper(0),
        errorsForUpstreamWrapper(0),
        outputCopiedToHost(false),
        errorsForUpstreamCopiedToHost(false),
        batchSize(0),
        allocatedSize(0) {
    if( inputImageSize == 0 ){
//        maker->net->print();
        throw runtime_error("Error: Dropout layer " + toString( layerIndex ) + ": input image size is 0" );
    }
    if( outputImageSize == 0 ){
//        maker->net->print();
        throw runtime_error("Error: Dropout layer " + toString( layerIndex ) + ": output image size is 0" );
    }
    dropoutPropagateImpl = DropoutPropagate::instance( cl, numPlanes, inputImageSize, dropRatio );
    dropoutBackpropImpl = DropoutBackprop::instance( cl, numPlanes, inputImageSize, dropRatio );
    multiplyBuffer = new MultiplyBuffer( cl, dropRatio );
}
VIRTUAL DropoutLayer::~DropoutLayer() {
    delete multiplyBuffer;
    delete dropoutPropagateImpl;
    delete dropoutBackpropImpl;
    if( maskWrapper != 0 ) {
        delete maskWrapper;
    }
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    if( masks != 0 ) {
        delete[] masks;
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
VIRTUAL std::string DropoutLayer::getClassName() const {
    return "DropoutLayer";
}
VIRTUAL void DropoutLayer::fortesting_setRandomSingleton( RandomSingleton *random ) {
    this->random = random;
}
VIRTUAL void DropoutLayer::setBatchSize( int batchSize ) {
//    cout << "DropoutLayer::setBatchSize" << endl;
    if( batchSize <= allocatedSize ) {
        this->batchSize = batchSize;
        return;
    }
    if( maskWrapper != 0 ) {
        delete maskWrapper;
    }
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    if( masks != 0 ) {
        delete[] masks;
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
    masks = new unsigned char[ getOutputSize() ];
    maskWrapper = cl->wrap( getOutputSize(), masks );
    output = new float[ getOutputSize() ];
    outputWrapper = cl->wrap( getOutputSize(), output );
    errorsForUpstream = new float[ previousLayer->getOutputSize() ];
    errorsForUpstreamWrapper = cl->wrap( previousLayer->getOutputSize(), errorsForUpstream );
    errorsForUpstreamWrapper->createOnDevice();
}
VIRTUAL int DropoutLayer::getOutputSize() {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL float *DropoutLayer::getOutput() {
    if( !outputCopiedToHost ) {
        outputWrapper->copyToHost();
        outputCopiedToHost = true;
    }
    return output;
}
VIRTUAL bool DropoutLayer::needsBackProp() {
    return previousLayer->needsBackProp(); // seems highly unlikely that we wouldnt have to backprop
                                           // but anyway, we dont have any weights ourselves
                                           // so just depends on upstream
}
VIRTUAL int DropoutLayer::getOutputSize() const {
//    int outputImageSize = inputImageSize / dropoutSize;
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL int DropoutLayer::getOutputImageSize() const {
    return outputImageSize;
}
VIRTUAL int DropoutLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int DropoutLayer::getPersistSize() const {
    return 0;
}
VIRTUAL bool DropoutLayer::providesGradInputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *DropoutLayer::getGradInputWrapper() {
    return errorsForUpstreamWrapper;
}
VIRTUAL bool DropoutLayer::hasOutputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *DropoutLayer::getOutputWrapper() {
    return outputWrapper;
}
VIRTUAL float *DropoutLayer::getGradInput() {
    return errorsForUpstream;
}
VIRTUAL ActivationFunction const *DropoutLayer::getActivationFunction() {
    return new LinearActivation();
}
//VIRTUAL void DropoutLayer::generateMasks() {
//    int totalInputLinearSize = getOutputSize();
////    int numBytes = (totalInputLinearSize+8-1)/8;
////    unsigned char *bitsField = new unsigned char[numBytes];
//    int idx = 0;
//    unsigned char thisByte = 0;
//    int bitsPacked = 0;
//    for( int i = 0; i < totalInputLinearSize; i++ ) {
//        //double value = ( (int)random() % 10000 ) / 20000.0f + 0.5f;
//        // 1 means we pass value through, 0 means we drop
//        // dropRatio is probability that mask value is 0 therefore
//        // so higher dropRatio => more likely to be 0
//        unsigned char bit = random->_uniform() <= dropRatio ? 0 : 1;
////        unsigned char bit = 0;
//        thisByte <<= 1;
//        thisByte |= bit;
//        bitsPacked++;
//        if( bitsPacked >= 8 ) {
//            masks[idx] = thisByte;
//            idx++;
//            bitsPacked = 0;
//        }
//    }
//}
VIRTUAL void DropoutLayer::generateMasks() {
    int totalInputLinearSize = getOutputSize();
    for( int i = 0; i < totalInputLinearSize; i++ ) {
        masks[i] = random->_uniform() <= dropRatio ? 0 : 1;
    }
}
VIRTUAL void DropoutLayer::propagate() {
    CLWrapper *upstreamOutputWrapper = 0;
    if( previousLayer->hasOutputWrapper() ) {
        upstreamOutputWrapper = previousLayer->getOutputWrapper();
    } else {
        float *upstreamOutput = previousLayer->getOutput();
        upstreamOutputWrapper = cl->wrap( previousLayer->getOutputSize(), upstreamOutput );
        upstreamOutputWrapper->copyToDevice();
    }

//    cout << "training: " << training << endl;
    if( training ) {
        // create new masks...
        generateMasks();
        maskWrapper->copyToDevice();
        dropoutPropagateImpl->propagate( batchSize, maskWrapper, upstreamOutputWrapper, outputWrapper );
    } else {
        // if not training, then simply skip the dropout bit, copy the buffers directly
        multiplyBuffer->multiply( getOutputSize(), upstreamOutputWrapper, outputWrapper );
    }
    if( !previousLayer->hasOutputWrapper() ) {
        delete upstreamOutputWrapper;
    }
}
VIRTUAL void DropoutLayer::backProp( float learningRate ) {
    // have no weights to backprop to, just need to backprop the errors

    CLWrapper *errorsWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if( nextLayer->providesGradInputWrapper() ) {
        errorsWrapper = nextLayer->getGradInputWrapper();
    } else {
        errorsWrapper = cl->wrap( getOutputSize(), nextLayer->getGradInput() );
        errorsWrapper->copyToDevice();
        weOwnErrorsWrapper = true;
    }
    dropoutBackpropImpl->backpropErrors( batchSize, maskWrapper, errorsWrapper, errorsForUpstreamWrapper );
    if( weOwnErrorsWrapper ) {
        delete errorsWrapper;
    }
}
VIRTUAL std::string DropoutLayer::asString() const {
    return "DropoutLayer{ dropRatio=" + toString(dropRatio) + " }";
}


