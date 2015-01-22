// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "NeuralNet.h"

#include "FullyConnectedLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

FullyConnectedLayer::FullyConnectedLayer( Layer *previousLayer, FullyConnectedMaker const*maker ) :
        Layer( previousLayer, maker ),
        numPlanes( maker->getNumPlanes() ),
        boardSize( maker->getBoardSize() ),
        fn( maker->getActivationFunction() ) {
    ConvolutionalMaker *convolutionalMaker = new ConvolutionalMaker( maker->net );
    convolutionalMaker->numFilters( numPlanes * boardSize * boardSize )
                      ->filterSize( previousLayer->getOutputBoardSize() )
                        ->biased( maker->_biased )
                        ->fn( maker->getActivationFunction() );
    convolutionalLayer = new ConvolutionalLayer( previousLayer, convolutionalMaker );
    delete convolutionalMaker;
}

VIRTUAL FullyConnectedLayer::~FullyConnectedLayer() {
    delete convolutionalLayer;
}
VIRTUAL void FullyConnectedLayer::setBatchSize( int batchSize ) {
    convolutionalLayer->previousLayer = this->previousLayer;
    convolutionalLayer->nextLayer = this->nextLayer;
    convolutionalLayer->setBatchSize( batchSize );
    this->batchSize = batchSize;
}
VIRTUAL int FullyConnectedLayer::getOutputBoardSize() const {
    return boardSize;
}
VIRTUAL int FullyConnectedLayer::getOutputPlanes() const {
    return numPlanes;
}
VIRTUAL int FullyConnectedLayer::getResultsSize() const {
    return convolutionalLayer->getResultsSize();
}
VIRTUAL float *FullyConnectedLayer::getResults() {
    return convolutionalLayer->getResults();
}
VIRTUAL float *FullyConnectedLayer::getErrorsForUpstream() {
    return convolutionalLayer->getErrorsForUpstream();
}
VIRTUAL bool FullyConnectedLayer::providesErrorsForUpstreamWrapper() const {
    return convolutionalLayer->providesErrorsForUpstreamWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getErrorsForUpstreamWrapper() {
    return convolutionalLayer->getErrorsForUpstreamWrapper();
}
VIRTUAL bool FullyConnectedLayer::hasResultsWrapper() const {
    return convolutionalLayer->hasResultsWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getResultsWrapper() {
    return convolutionalLayer->getResultsWrapper();
}
VIRTUAL void FullyConnectedLayer::propagate() {
    convolutionalLayer->propagate();
}
VIRTUAL void FullyConnectedLayer::backProp( float learningRate ) {
    convolutionalLayer->backProp( learningRate );
}
VIRTUAL std::string FullyConnectedLayer::asString() const {
    return "FullyConnectedLayer{ numPlanes=" + toString( numPlanes ) + " boardSize=" + toString( boardSize ) + " " + fn->getDefineName() + " }";
}

