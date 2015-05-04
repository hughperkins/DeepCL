// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "net/NeuralNet.h"
#include "util/stringhelper.h"

#include "activate/ActivationLayer.h"
#include "activate/ActivationMaker.h"
#include "activate/ActivationForward.h"
#include "activate/ActivationBackward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

ActivationLayer::ActivationLayer( EasyCL *cl, Layer *previousLayer, ActivationMaker *maker ) :
        Layer( previousLayer, maker ),
        numPlanes ( previousLayer->getOutputPlanes() ),
        inputImageSize( previousLayer->getOutputImageSize() ),
        outputImageSize( previousLayer->getOutputImageSize() ),
        fn( maker->_activationFunction ),
        cl( cl ),
        output(0),
        gradInput(0),
        outputWrapper(0),
        gradInputWrapper(0),
        outputCopiedToHost(false),
        gradInputCopiedToHost(false),
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
    activationForwardImpl = ActivationForward::instance( cl, numPlanes, inputImageSize, fn );
    activationBackpropImpl = ActivationBackward::instance( cl, numPlanes, inputImageSize, fn );
}
VIRTUAL ActivationLayer::~ActivationLayer() {
    delete activationForwardImpl;
    delete activationBackpropImpl;
    if( outputWrapper != 0 ) {
        delete outputWrapper;
    }
    if( output != 0 ) {
        delete[] output;
    }
    if( gradInputWrapper != 0 ) {
        delete gradInputWrapper;
    }
    if( gradInput != 0 ) {
        delete[] gradInput;
    }
}
VIRTUAL std::string ActivationLayer::getClassName() const {
    return "ActivationLayer";
}
VIRTUAL float ActivationLayer::getOutput( int n, int plane, int row, int col ) {
    int index = ( ( n
        * numPlanes + plane )
        * outputImageSize + row )
        * outputImageSize + col;
    return output[ index ];
}
VIRTUAL void ActivationLayer::printOutput() {
//    float const*output = getOutput();
//    int outPlanes = getOutputPlanes();
//    int outputSize = getOutputImageSize();
    std::cout << "  outputs: " << std::endl;
    getOutput();
// output are organized like [imageid][filterid][row][col]
    for( int n = 0; n < std::min( 5, batchSize ); n++ ) {
        std::cout << "    n: " << n << std::endl;
        for( int plane = 0; plane < std::min(5, numPlanes ); plane++ ) {
            if( numPlanes > 1 ) std::cout << "      plane " << plane << std::endl;
            if( outputImageSize == 1 ) {
                 std::cout << "        " << getOutput(n, plane, 0, 0 ) << std::endl;
            } else {
                for( int i = 0; i < std::min(5, outputImageSize); i++ ) {
                    std::cout << "      ";
                    for( int j = 0; j < std::min(5, outputImageSize); j++ ) {
                        std::cout << getOutput( n, plane, i, j ) << " ";
                    }
                    if( outputImageSize > 5 ) std::cout << " ... ";
                    std::cout << std::endl;
                }
                if( outputImageSize > 5 ) std::cout << " ... " << std::endl;
            }
            if( numPlanes > 5 ) std::cout << " ... other planes ... " << std::endl;
        }
        if( batchSize > 5 ) std::cout << " ... other n ... " << std::endl;
    }
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
    outputWrapper->createOnDevice();
    gradInput = new float[ previousLayer->getOutputSize() ];
    gradInputWrapper = cl->wrap( previousLayer->getOutputSize(), gradInput );
    gradInputWrapper->createOnDevice();
}
VIRTUAL int ActivationLayer::getOutputSize() {
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL float *ActivationLayer::getOutput() {
    if( !outputCopiedToHost ) {
        outputWrapper->copyToHost();
        outputCopiedToHost = true;
    }
//    cout << "getOutput output[0] " << output[0] << " output[1] " << output[1] << endl;
    return output;
}
VIRTUAL bool ActivationLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL int ActivationLayer::getOutputSize() const {
//    int outputImageSize = inputImageSize / poolingSize;
    return batchSize * numPlanes * outputImageSize * outputImageSize;
}
VIRTUAL int ActivationLayer::getOutputCubeSize() const {
    return numPlanes * outputImageSize * outputImageSize;
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
    return gradInputWrapper;
}
VIRTUAL bool ActivationLayer::hasOutputWrapper() const {
    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getOutputWrapper() {
    return outputWrapper;
}
VIRTUAL int ActivationLayer::getWeightsSize() const {
    return 0;
}
VIRTUAL int ActivationLayer::getBiasSize() const {
    return 0;
}
VIRTUAL float *ActivationLayer::getGradInput() {
    if( !gradInputCopiedToHost ) {
        gradInputWrapper->copyToHost();
        gradInputCopiedToHost = true;
    }
    return gradInput;
}
VIRTUAL ActivationFunction const *ActivationLayer::getActivationFunction() {
    return fn;
}
VIRTUAL void ActivationLayer::forward() {
    CLWrapper *inputWrapper = 0;
    if( previousLayer->hasOutputWrapper() ) {
        inputWrapper = previousLayer->getOutputWrapper();
    } else {
        float *input = previousLayer->getOutput();
        inputWrapper = cl->wrap( previousLayer->getOutputSize(), input );
        inputWrapper->copyToDevice();
    }
    activationForwardImpl->forward( batchSize, inputWrapper, outputWrapper );
    outputCopiedToHost = false;
    if( !previousLayer->hasOutputWrapper() ) {
        delete inputWrapper;
    }
}
VIRTUAL void ActivationLayer::backward() {
    // have no weights to backprop to, just need to backprop the errors

//    CLWrapper *imagesWrapper = 0;
//    if( previousLayer->hasOutputWrapper() ) {
//        imagesWrapper = previousLayer->getOutputWrapper();
//    } else {
//        imagesWrapper = cl->wrap( previousLayer->getOutputSize(), previousLayer->getOutput() );
//        imagesWrapper->copyToDevice();
//    }

    CLWrapper *gradOutputWrapper = 0;
    bool weOwnGradOutputWrapper = false;
    if( nextLayer->providesGradInputWrapper() ) {
        gradOutputWrapper = nextLayer->getGradInputWrapper();
    } else {
        gradOutputWrapper = cl->wrap( getOutputSize(), nextLayer->getGradInput() );
        gradOutputWrapper->copyToDevice();
        weOwnGradOutputWrapper = true;
    }

    activationBackpropImpl->backward( batchSize, outputWrapper, gradOutputWrapper, gradInputWrapper );
    gradInputCopiedToHost = false;

//    if( !previousLayer->hasOutputWrapper() ) {
//        delete imagesWrapper;
//    }
    if( weOwnGradOutputWrapper ) {
        delete gradOutputWrapper;
    }
}
VIRTUAL std::string ActivationLayer::asString() const {
    return "ActivationLayer{ " + fn->getDefineName() + " }";
}
VIRTUAL int ActivationLayer::getPersistSize() const {
    // no weights, so:
    return 0;
}

