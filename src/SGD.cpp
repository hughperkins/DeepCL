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
#include "SGDStateMaker.h"
#include "SGDState.h"
#include "SGD.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


VIRTUAL SGD::~SGD() {
    delete kernel;
}
VIRTUAL void SGD::setMomentum( float momentum ) {
    this->momentum = momentum;
}
VIRTUAL std::string SGD::asString() {
    return "SGD{ learningRate=" + toString( learningRate ) + ", momentum=" + 
        toString( momentum ) + " }";
}
VIRTUAL void SGD::updateWeights( CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
        SGDState *trainerState ) {
    
}
VIRTUAL void SGD::learn( float *input, float *expectedOutput ) { // learns one batch, including updating weights
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
        updateWeights( layer->getWeightsWrapper(), layer->getGradWeightsWrapper(), 
            dynamic_cast< SGDState * >( layer->getTrainerState() ) );
        if( layer->biased() ) {
            updateWeights( layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                dynamic_cast< SGDState * >( layer->getBiasTrainerState() ) );
        }
    }
}
SGD::SGD( OpenCLHelper *cl, NeuralNet *net ) :
        Trainer( cl, net ),
        kernel( 0 ) {
    SGDStateMaker stateMaker;
    // go through network layers, and assign SGD objects (should probably rename these sometime somehow)
    for( int layerIdx = 0; layerIdx < net->getNumLayers(); layerIdx++ ) {
        Layer *layer = net->getLayer( layerIdx );
        if( layer->needsTrainerState() ) {
            layer->setTrainerState( &stateMaker );
        }
    }
    string options = "";
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "cl/SGD.cl", "updateWeights", 'options' )
    // ]]]
    // generated using cog, from cl/SGD.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void updateWeights(\n" 
    "        const int N,\n" 
    "        const float learningRate,\n" 
    "        const float momentum,\n" 
    "        global float *lastUpdate,\n" 
    "        global const float *gradWeights,\n" 
    "        global float *weights\n" 
    "            ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    // first update the update\n" 
    "    lastUpdate[globalId] =\n" 
    "        momentum * lastUpdate[globalId]\n" 
    "        - learningRate * gradWeights[globalId];\n" 
    "    // now update the weight\n" 
    "    weights[globalId] += lastUpdate[globalId];\n" 
    "    // thats it... :-)\n" 
    "}\n" 
    "\n" 
    "";
    kernel = cl->buildKernelFromString( kernelSource, "updateWeights", options, "cl/SGD.cl" );
    // [[[end]]]
}

