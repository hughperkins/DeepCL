// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "NetTestHelper.h"

#undef STATIC
#define STATIC
#undef VIRTUAL
#define VIRTUAL

PUBLIC STATIC void NetTestHelper::printWeightsAsCode( Layer *layer ) {
    int layerIndex = layer->getLayerIndex();
    std::cout << "float weights" << layerIndex << "[] = {";
    const int numWeights = layer->getWeightsSize();
    float const*weights = layer->getWeights();
    for( int i = 0; i < numWeights; i++ ) {
        std::cout << weights[i] << "f";
        if( i < numWeights - 1 ) std::cout << ", ";
        if( i > 0 && i % 20 == 0 ) std::cout << std::endl;
    }
    std::cout << "};" << std::endl;
}
PUBLIC STATIC void NetTestHelper::printBiasAsCode( Layer *layer ) {
    int layerIndex = layer->getLayerIndex();
    std::cout << "float bias" << layerIndex << "[] = {";
    const int numBias = layer->getBiasSize();
    float const*bias = layer->getBias();
    for( int i = 0; i < numBias; i++ ) {
        std::cout << bias[i] << "f";
        if( i < numBias - 1 ) std::cout << ", ";
        if( i > 0 && i % 20 == 0 ) std::cout << std::endl;
    }
    std::cout << "};" << std::endl;
//        std::cout << netObjectName << "->layers[" << layerIndex << "]->weights[
}
PUBLIC STATIC void NetTestHelper::printWeightsAsCode(NeuralNet *net) {
    int numLayers = net->getNumLayers();
    for( int layerIdx = 1; layerIdx < numLayers; layerIdx++ ) {
        Layer *layer = net->getLayer( layerIdx );
        int persistSize = layer->getPersistSize();
        if( persistSize == 0 ) {
            continue;
        }
        printWeightsAsCode( layer );
    }
}
PUBLIC STATIC void NetTestHelper::printBiasAsCode(NeuralNet *net) {
    int numLayers = net->getNumLayers();
    for( int layerIdx = 1; layerIdx < numLayers; layerIdx++ ) {
        Layer *layer = net->getLayer( layerIdx );
        int persistSize = layer->getPersistSize();
        if( persistSize == 0 ) {
            continue;
        }
        printBiasAsCode( layer );
    }
}

