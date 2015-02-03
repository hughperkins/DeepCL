// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include "NeuralNet.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "InputLayer.h"
#include "SoftMaxLayer.h"
#include "SquareLossLayer.h"
#include "CrossEntropyLoss.h"
#include "PoolingLayer.h"
#include "NormalizationLayer.h"
#include "RandomPatches.h"
#include "RandomTranslations.h"

#include "LayerMaker.h"

using namespace std;

Layer *LayerMaker::insert() {
    Layer *layer = net->addLayer( this );
//    delete this;
    return layer;
}
Layer *FullyConnectedMaker::insert() {
    if( _numPlanes == 0 ) {
        throw runtime_error("Must provide ->planes(planes)");
    }
    if( _boardSize == 0 ) {
        throw runtime_error("Must provide ->boardSize(boardSize)");
    }
//    Layer *layer = net->addFullyConnected( _numPlanes, _boardSize, _biased, _activationFunction );
    Layer *layer = net->addLayer( this );
//    delete this;
    return layer;
}
template< typename T > Layer *InputLayerMaker<T>::insert() {
    if( _numPlanes == 0 ) {
        throw runtime_error("Must provide ->numPlanes(planes)");
    }
    if( _boardSize == 0 ) {
        throw runtime_error("Must provide ->boardSize(boardSize)");
    }
    Layer *layer = net->addLayer( this );
//    delete this;
    return layer;
}
//Layer *NormalizationLayerMaker::insert() {
//    Layer *layer = net->addLayer( this );
//    delete this;
//    return layer;
//}
Layer *ConvolutionalMaker::insert() {
    if( _numFilters == 0 ) {
        throw runtime_error("Must provide ->numFilters(numFilters)");
    }
    if( _filterSize == 0 ) {
        throw runtime_error("Must provide ->filterSize(filterSize)");
    }
    Layer *layer = net->addLayer( this );
//    delete this;
    return layer;
}

template< typename T>
LayerMaker *InputLayerMaker<T>::clone( NeuralNet *net, Layer *previousLayer ) const {
    InputLayerMaker<T> *maker = new InputLayerMaker<T>( net );
    maker->_numPlanes = _numPlanes;
    maker->_boardSize = _boardSize;
    return maker;
}
LayerMaker *NormalizationLayerMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    NormalizationLayerMaker *maker = new NormalizationLayerMaker( net, previousLayer );
    maker->_translate = _translate;
    maker->_scale = _scale;
    return maker;
}
LayerMaker *RandomPatchesMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    RandomPatchesMaker *maker = new RandomPatchesMaker( net, previousLayer );
    maker->_patchSize = _patchSize;
    return maker;
}
LayerMaker *RandomTranslationsMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    RandomTranslationsMaker *maker = new RandomTranslationsMaker( net, previousLayer );
    maker->_translateSize = _translateSize;
    return maker;
}
LayerMaker *PoolingMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    PoolingMaker *maker = new PoolingMaker( net, previousLayer );
    maker->_poolingSize = _poolingSize;
    maker->_padZeros = _padZeros;
    return maker;
}
LayerMaker *FullyConnectedMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    FullyConnectedMaker *maker = new FullyConnectedMaker( net, previousLayer );
    maker->_numPlanes = _numPlanes;
    maker->_boardSize = _boardSize;
    maker->_biased = _biased;
    maker->_activationFunction = _activationFunction;
    return maker;
}
LayerMaker *SquareLossMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    SquareLossMaker *maker = new SquareLossMaker( net, previousLayer );
    return maker;
}
LayerMaker *CrossEntropyLossMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    CrossEntropyLossMaker *maker = new CrossEntropyLossMaker( net, previousLayer );
    return maker;
}
LayerMaker *SoftMaxMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    SoftMaxMaker *maker = new SoftMaxMaker( net, previousLayer );
    maker->_perPlane = _perPlane;
    return maker;
}
LayerMaker *ConvolutionalMaker::clone( NeuralNet *net, Layer *previousLayer ) const {
    ConvolutionalMaker *maker = new ConvolutionalMaker( net, previousLayer );
    maker->_numFilters = _numFilters;
    maker->_filterSize = _filterSize;
    maker->_padZeros = _padZeros;
    maker->_biased = _biased;
    maker->_activationFunction = _activationFunction;
    return maker;
}

Layer *FullyConnectedMaker::instance() const {
    Layer *layer = new FullyConnectedLayer( previousLayer, this );
    return layer;
}
Layer *SquareLossMaker::instance() const {
    SquareLossLayer *layer = new SquareLossLayer( previousLayer, this );
    return layer;
}
Layer *CrossEntropyLossMaker::instance() const {
    CrossEntropyLoss *layer = new CrossEntropyLoss( previousLayer, this );
    return layer;
}
Layer *SoftMaxMaker::instance() const {
    Layer *layer = new SoftMaxLayer( previousLayer, this );
    return layer;
}
Layer *PoolingMaker::instance() const {
//    if( previousLayer->getOutputBoardSize() % 2 != 0 ) {
//        throw std::runtime_error("For now, pooling layer only handles inputboardsizes with even number.  You specified: " + toString( previousLayer->getOutputBoardSize() ) );
//    }
    Layer *layer = new PoolingLayer( previousLayer, this );
    return layer;
}
Layer *ConvolutionalMaker::instance() const {
    Layer *layer = new ConvolutionalLayer( previousLayer, this );
    return layer;
}
template< typename T > Layer *InputLayerMaker<T>::instance() const {
    Layer *layer = new InputLayer<T>( 0, this );
    return layer;
}
Layer *NormalizationLayerMaker::instance() const {
    Layer *layer = new NormalizationLayer( previousLayer, this );
    return layer;
}
Layer *RandomPatchesMaker::instance() const {
    Layer *layer = new RandomPatches( previousLayer, this );
    return layer;
}
Layer *RandomTranslationsMaker::instance() const {
    Layer *layer = new RandomTranslations( previousLayer, this );
    return layer;
}

int ConvolutionalMaker::getOutputBoardSize() const {
    if( previousLayer == 0 ) {
        throw std::runtime_error("convolutional network must be attached to a parent layer");
    }
    int evenPadding = _filterSize % 2 == 0 ? 1 : 0;
    int boardSize = _padZeros ? previousLayer->getOutputBoardSize() + evenPadding : previousLayer->getOutputBoardSize() - _filterSize + 1;
    return boardSize;
}

int LossLayerMaker::getOutputBoardSize() const {
    return previousLayer->getOutputBoardSize();
}
int LossLayerMaker::getOutputPlanes() const {
    return previousLayer->getOutputPlanes();
}
int LossLayerMaker::getBiased() const {
    return previousLayer->getBiased();
}
int PoolingMaker::getOutputBoardSize() const {
    return previousLayer->getOutputBoardSize() / _poolingSize;
}
int PoolingMaker::getOutputPlanes() const {
    return previousLayer->getOutputPlanes();
}
int PoolingMaker::getBiased() const {
    return false;
}
int NormalizationLayerMaker::getOutputBoardSize() const {
    return previousLayer->getOutputBoardSize();
}
int NormalizationLayerMaker::getOutputPlanes() const {
    return previousLayer->getOutputPlanes();
}
int NormalizationLayerMaker::getBiased() const {
    return false;
}
int RandomPatchesMaker::getOutputBoardSize() const {
    return _patchSize;
}
int RandomPatchesMaker::getOutputPlanes() const {
    return previousLayer->getOutputPlanes();
}
int RandomPatchesMaker::getBiased() const {
    return false;
}

int RandomTranslationsMaker::getOutputBoardSize() const {
    return previousLayer->getOutputBoardSize();
}
int RandomTranslationsMaker::getOutputPlanes() const {
    return previousLayer->getOutputPlanes();
}
int RandomTranslationsMaker::getBiased() const {
    return false;
}

template class InputLayerMaker<float>;
template class InputLayerMaker<unsigned char>;

