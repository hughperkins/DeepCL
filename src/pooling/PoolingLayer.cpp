// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "PoolingMaker.h"
#include "PoolingLayer.h"
#include "PoolingForward.h"
#include "PoolingBackward.h"

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingLayer::PoolingLayer( EasyCL *cl, Layer *previousLayer, PoolingMaker *maker ) :
        Layer( previousLayer, maker ),
        padZeros( maker->_padZeros ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputImageSize( previousLayer->getOutputImageSize() ),
        poolingSize( maker->_poolingSize ),
        outputImageSize( maker->_padZeros ? ( previousLayer->getOutputImageSize() + maker->_poolingSize - 1 ) / maker->_poolingSize : previousLayer->getOutputImageSize() / maker->_poolingSize ),
        cl( cl ),
        output(0),
        selectors(0),
        gradInput(0),
        outputWrapper(0),
        selectorsWrapper(0),
        gradInputWrapper(0),
        outputCopiedToHost(false),
        gradInputCopiedToHost(false),
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
    poolingForwardImpl = PoolingForward::instance( cl, padZeros, numPlanes, inputImageSize, poolingSize );
    poolingBackpropImpl = PoolingBackward::instance( cl, padZeros, numPlanes, inputImageSize, poolingSize );
}
VIRTUAL PoolingLayer::~PoolingLayer() {
    delete poolingForwardImpl;
    delete poolingBackpropImpl;
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    if( output != 0 ) {
        delete[] output;
    }
    if( selectorsWrapper != 0 ) {
        delete selectorsWrapper;
    }
    if( selectors != 0 ) {
        delete[] selectors;
    }
    if( gradInputWrapper != 0 ) {
        delete gradInputWrapper;
    }
    if( gradInput != 0 ) {
        delete[] gradInput;
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
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    if( output != 0 ) {
        delete[] output;
    }
    if( selectorsWrapper != 0 ) {
        delete selectorsWrapper;
    }
    if( selectors != 0 ) {
        delete[] selectors;
    }
    if( gradInputWrapper != 0 ) {
        delete gradInputWrapper;
    }
    if( gradInput != 0 ) {
        delete[] gradInput;
    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
    output = new float[ getOutputSize() ];
    outputWrapper = cl->wrap( getOutputSize(), output );
    selectors = new int[ getOutputSize() ];
    selectorsWrapper = cl->wrap( getOutputSize(), selectors );
    gradInput = new float[ previousLayer->getOutputSize() ];
    gradInputWrapper = cl->wrap( previousLayer->getOutputSize(), gradInput );
    gradInputWrapper->createOnDevice();
}
VIRTUAL int PoolingLayer::getOutputSize() {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL float *PoolingLayer::getOutput() {
    if( !outputCopiedToHost ) {
        outputWrapper->copyToHost();
        outputCopiedToHost = true;
    }
    return output;
}
VIRTUAL bool PoolingLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL int PoolingLayer::getOutputSize() const {
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
VIRTUAL bool PoolingLayer::providesGradInputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *PoolingLayer::getGradInputWrapper() {
    return gradInputWrapper;
}
VIRTUAL bool PoolingLayer::hasOutputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *PoolingLayer::getOutputWrapper() {
    return outputWrapper;
}
VIRTUAL float *PoolingLayer::getGradInput() {
    return gradInput;
}
VIRTUAL ActivationFunction const *PoolingLayer::getActivationFunction() {
    //return previousLayer->getActivationFunction(); // I guess???
    return new LinearActivation();
}
VIRTUAL void PoolingLayer::forward() {
    CLWrapper *upstreamOutputWrapper = 0;
    if( previousLayer->hasOutputWrapper() ) {
        upstreamOutputWrapper = previousLayer->getOutputWrapper();
    } else {
        float *upstreamOutput = previousLayer->getOutput();
        upstreamOutputWrapper = cl->wrap( previousLayer->getOutputSize(), upstreamOutput );
        upstreamOutputWrapper->copyToDevice();
    }
    poolingForwardImpl->forward( batchSize, upstreamOutputWrapper, selectorsWrapper, outputWrapper );
    if( !previousLayer->hasOutputWrapper() ) {
        delete upstreamOutputWrapper;
    }

//    cout << "PoolingLayer::forward() selectors after forward: " << endl;
//    for( int i = 0; i < outputImageSize; i++ ) {
//        for( int j = 0; j < outputImageSize; j++ ) {
//            cout << selectors[ i * outputImageSize + j ] << " ";
//        }
//        cout << endl;
//    }

//    cout << "PoolingLayer::forward() selectorsWrapper after forward: " << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputImageSize, outputImageSize );
}
VIRTUAL void PoolingLayer::backward() {
    // have no weights to backprop to, just need to backprop the errors

    CLWrapper *gradOutputWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if( nextLayer->providesGradInputWrapper() ) {
        gradOutputWrapper = nextLayer->getGradInputWrapper();
    } else {
        gradOutputWrapper = cl->wrap( getOutputSize(), nextLayer->getGradInput() );
        gradOutputWrapper->copyToDevice();
        weOwnErrorsWrapper = true;
    }

//    cout << "PoolingLayer::backward selectorsWrapper:" << endl;
//    PrintBuffer::printInts( cl, selectorsWrapper, outputImageSize, outputImageSize );

//    int *selectors = reinterpret_cast< int * >( selectorsWrapper->getHostArray() );
//    cout << "PoolingLayer::backward selectors before copy to host:" << endl;
//    for( int i = 0; i < outputImageSize; i++ ) {
//        for( int j = 0; j < outputImageSize; j++ ) {
//            cout << " " << selectors[i * outputImageSize + j];
//        }
//        cout << endl;
//    }
//    selectorsWrapper->copyToHost();
//    cout << "PoolingLayer::backward selectors after copy to host:" << endl;
//    for( int i = 0; i < outputImageSize; i++ ) {
//        for( int j = 0; j < outputImageSize; j++ ) {
//            cout << " " << selectors[i * outputImageSize + j];
//        }
//        cout << endl;
//    }
//    selectorsWrapper->copyToDevice();

//    selectorsWrapper->copyToHost();

    poolingBackpropImpl->backward( batchSize, gradOutputWrapper, selectorsWrapper, gradInputWrapper );

//    gradInputWrapper->copyToHost();
//    float *gradInput = reinterpret_cast< float * >( gradInputWrapper->getHostArray() );
//    cout << "gradInput:" << endl;
//    for( int i = 0; i < inputImageSize; i++ ) {
//        for( int j = 0; j < inputImageSize; j++ ) {
////            cout << " " << gradInput[i * inputImageSize + j];
//            if( gradInput[i * inputImageSize + j] != 0 ) {
//                cout << " *";
//            } else {
//                cout << " .";
//            }
//        }
//        cout << endl;
//    }

    if( weOwnErrorsWrapper ) {
        delete gradOutputWrapper;
    }
}
VIRTUAL std::string PoolingLayer::asString() const {
    return "PoolingLayer{ inputPlanes=" + toString(numPlanes) + " inputImageSize=" + toString(inputImageSize) + " poolingSize=" + toString( poolingSize ) + " }";
}


