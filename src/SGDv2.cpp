// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "stringhelper.h"
#include "NeuralNet.h"
#include "Layer.h"
#include "LossLayer.h"
#include "SGDMaker.h"
#include "SGDv2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


SGDv2::SGDv2( OpenCLHelper *cl, NeuralNet *net ) :
        Trainerv2( cl, net ) {
    SGDMaker maker;
    // go through network layers, and assign SGD objects (should probably rename these sometime somehow)
    for( int layerIdx = 0; layerIdx < net->getNumLayers(); layerIdx++ ) {
        Layer *layer = net->getLayer( layerIdx );
        if( layer->needsTrainer() ) {
            layer->setTrainer( &maker );
        }
    }    
}
VIRTUAL SGDv2::~SGDv2() {
}
VIRTUAL void SGDv2::setMomentum( float momentum ) {
    this->momentum = momentum;
}
VIRTUAL std::string SGDv2::asString() {
    return "SGDv2{ learningRate=" + toString( learningRate ) + ", momentum=" + 
        toString( momentum ) + " }";
}
VIRTUAL void SGDv2::learn( float *input, float *expectedOutput ) { // learns one batch, including updating weights
                                  // doesnt have to think about running multiple batches,
                                  // or loading data, or anything like that
    // net->calcGrad();
    net->forward( input );
    int numLayers = net->getNumLayers();
    LossLayer *lossLayer = dynamic_cast< LossLayer * >( net->getLastLayer() );
    if( lossLayer == 0 ) {
        throw runtime_error( "last layer of net should be a LossLayer class" );
    }
    lossLayer->calcGradInput( expectedOutput );
    for( int layerIdx = numLayers - 2; layerIdx > 0; layerIdx-- ) {
        Layer *layer = net->getLayer( layerIdx );
        layer->backward();
    }
}

