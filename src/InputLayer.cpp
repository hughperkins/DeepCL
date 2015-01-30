// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "InputLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

InputLayer::InputLayer( Layer *previousLayer, InputLayerMaker const*maker ) :
       Layer( previousLayer, maker ),
    batchSize(0),
    output(0),
    outputPlanes( maker->getOutputPlanes() ),
    outputBoardSize( maker->getOutputBoardSize() ) {
}
VIRTUAL InputLayer::~InputLayer() {
}
VIRTUAL float *InputLayer::getResults() {
    return output;
}
VIRTUAL ActivationFunction const *InputLayer::getActivationFunction() {
    return new LinearActivation();
}
VIRTUAL bool InputLayer::needsBackProp() {
    return false;
}
VIRTUAL void InputLayer::printOutput() const {
    if( output == 0 ) {
         return;
    }
    for( int n = 0; n < std::min(5,batchSize); n++ ) {
        std::cout << "InputLayer n " << n << ":" << std::endl;
        for( int plane = 0; plane < std::min( 5, outputPlanes); plane++ ) {
            if( outputPlanes > 1 ) std::cout << "    plane " << plane << ":" << std::endl;
            for( int i = 0; i < std::min(5, outputBoardSize); i++ ) {
                std::cout << "      ";
                for( int j = 0; j < std::min(5, outputBoardSize); j++ ) {
                    std::cout << getResult( n, plane, i, j ) << " ";
//results[
//                            n * numPlanes * boardSize*boardSize +
//                            plane*boardSize*boardSize +
//                            i * boardSize +
//                            j ] << " ";
                }
                if( outputBoardSize > 5 ) std::cout << " ... ";
                std::cout << std::endl;
            }
            if( outputBoardSize > 5 ) std::cout << " ... " << std::endl;
        }
        if( outputPlanes > 5 ) std::cout << "   ... other planes ... " << std::endl;
    }
    if( batchSize > 5 ) std::cout << "   ... other n ... " << std::endl;
}
VIRTUAL void InputLayer::print() const {
    printOutput();
}
void InputLayer::in( float const*images ) {
//        std::cout << "InputLayer::in()" << std::endl;
    this->output = (float*)images;
//        this->batchStart = batchStart;
//        this->batchEnd = batchEnd;
//        print();
}
VIRTUAL bool InputLayer::needErrorsBackprop() {
    return false;
}
VIRTUAL void InputLayer::setBatchSize( int batchSize ) {
//        std::cout << "inputlayer setting batchsize " << batchSize << std::endl;
    
    this->batchSize = batchSize;
}
VIRTUAL void InputLayer::propagate() {
}
VIRTUAL void InputLayer::backPropErrors( float learningRate, float const *errors ) {
}
VIRTUAL int InputLayer::getOutputBoardSize() const {
    return outputBoardSize;
}
VIRTUAL int InputLayer::getOutputPlanes() const {
    return outputPlanes;
}
VIRTUAL int InputLayer::getOutputCubeSize() const {
    return outputPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL int InputLayer::getResultsSize() const {
    return batchSize * getOutputCubeSize();
}
VIRTUAL std::string InputLayer::toString() {
    return std::string("") + "InputLayer { outputPlanes " + ::toString( outputPlanes ) + " outputBoardSize " +  ::toString( outputBoardSize ) + " }";
}
VIRTUAL std::string InputLayer::asString() const {
    return std::string("") + "InputLayer { outputPlanes " + ::toString( outputPlanes ) + " outputBoardSize " +  ::toString( outputBoardSize ) + " }";
}

//ostream &operator<<( ostream &os, InputLayer &layer ) {
//    os << "InputLayer { outputPlanes " << layer.outputPlanes << " outputBoardSize " << layer.outputBoardSize << " }";
//    return os;
//}
//ostream &operator<<( ostream &os, InputLayer const*layer ) {
//    os << "InputLayer { outputPlanes " << layer->outputPlanes << " outputBoardSize " << layer->outputBoardSize << " }";
//    return os;
//}

