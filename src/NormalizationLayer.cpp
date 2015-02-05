// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NormalizationLayerMaker.h"

#include "NormalizationLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

NormalizationLayer::NormalizationLayer( Layer *previousLayer, NormalizationLayerMaker const*maker ) :
       Layer( previousLayer, maker ),
    batchSize(0),
    allocatedSize(0),
    results(0),
    outputPlanes( previousLayer->getOutputPlanes() ),
    outputBoardSize( previousLayer->getOutputBoardSize() ),
    translate( maker->_translate ),
    scale( maker->_scale ) {
}
VIRTUAL NormalizationLayer::~NormalizationLayer() {
    if( results != 0 ) {
        delete[] results;
    }
}
VIRTUAL float *NormalizationLayer::getResults() {
    return results;
}
VIRTUAL ActivationFunction const *NormalizationLayer::getActivationFunction() {
    return new LinearActivation();
}
VIRTUAL bool NormalizationLayer::needsBackProp() {
    return previousLayer->needsBackProp();
}
VIRTUAL void NormalizationLayer::printOutput() const {
    if( results == 0 ) {
         return;
    }
    for( int n = 0; n < std::min(5,batchSize); n++ ) {
        std::cout << "NormalizationLayer n " << n << ":" << std::endl;
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
VIRTUAL void NormalizationLayer::print() const {
    printOutput();
}
VIRTUAL bool NormalizationLayer::needErrorsBackprop() {
    return false;
}
VIRTUAL void NormalizationLayer::setBatchSize( int batchSize ) {
    if( batchSize <= allocatedSize ) {
        this->batchSize = batchSize;
        return;
    }
    if( results != 0 ) {
        delete[] results;
    }
    this->batchSize = batchSize;
    this->allocatedSize = allocatedSize;
    results = new float[ getResultsSize() ];
}
VIRTUAL void NormalizationLayer::propagate() {
    int totalLinearLength = getResultsSize();
    float *upstreamResults = previousLayer->getResults();
    for( int i = 0; i < totalLinearLength; i++ ) {
        results[i] = ( upstreamResults[i] + translate ) * scale;
    }
}
VIRTUAL void NormalizationLayer::backPropErrors( float learningRate, float const *errors ) {
  // do nothing...
}
VIRTUAL int NormalizationLayer::getOutputBoardSize() const {
    return outputBoardSize;
}
VIRTUAL int NormalizationLayer::getOutputPlanes() const {
    return outputPlanes;
}
VIRTUAL int NormalizationLayer::getOutputCubeSize() const {
    return outputPlanes * outputBoardSize * outputBoardSize;
}
VIRTUAL int NormalizationLayer::getResultsSize() const {
    return batchSize * getOutputCubeSize();
}
VIRTUAL std::string NormalizationLayer::toString() {
    return toString();
}
VIRTUAL std::string NormalizationLayer::asString() const {
    return std::string("") + "NormalizationLayer { outputPlanes=" + ::toString( outputPlanes ) + " outputBoardSize=" +  ::toString( outputBoardSize ) + " translate=" + ::toString( translate ) + 
        " scale=" + ::toString( scale ) + " }";
}


