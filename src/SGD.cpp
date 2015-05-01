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
#include "IAcceptsLabels.h"

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
    int numWeights = trainerState->numWeights;
    CLWrapper *lastUpdateWrapper = trainerState->lastUpdateWrapper;
    kernel  ->in( numWeights )
            ->in( learningRate )
            ->in( momentum )
            ->inout( lastUpdateWrapper )
            ->in( gradWeightsWrapper )
            ->inout( weightsWrapper );
    int globalSize = numWeights;
    int workgroupSize = 64;
    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();
}
VIRTUAL void SGD::train( NeuralNet *net, float const*input, float const*expectedOutput ) {
//VIRTUAL void SGD::learn( float *input, float *expectedOutput ) { // learns one batch, including updating weights
                                  // doesnt have to think about running multiple batches,
                                  // or loading data, or anything like that
    // net->calcGrad();
    bindState( net );
    net->forward( input );
    int numLayers = net->getNumLayers();
    LossLayer *lossLayer = dynamic_cast< LossLayer * >( net->getLastLayer() );
    if( lossLayer == 0 ) {
        throw runtime_error( "last layer of net should be a LossLayer class" );
    }
    lossLayer->calcGradInput( expectedOutput );
    for( int layerIdx = numLayers - 2; layerIdx > 0; layerIdx-- ) {
        Layer *layer = net->getLayer( layerIdx );
        if( !layer->needsBackProp() ) {
            break;
        }
        layer->backward();
        if( layer->needsTrainerState() ) {
            updateWeights( layer->getWeightsWrapper(), layer->getGradWeightsWrapper(), 
                dynamic_cast< SGDState * >( layer->getTrainerState() ) );
            if( layer->biased() ) {
                updateWeights( layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< SGDState * >( layer->getBiasTrainerState() ) );
            }
        }
    }
}
VIRTUAL void SGD::trainFromLabels( NeuralNet *net, float const*input, int const*labels ) {
//VIRTUAL void SGD::learn( float *input, float *expectedOutput ) { // learns one batch, including updating weights
                                  // doesnt have to think about running multiple batches,
                                  // or loading data, or anything like that
    // net->calcGrad();
    bindState( net );
    net->forward( input );
    int numLayers = net->getNumLayers();
    IAcceptsLabels *lossLayer = dynamic_cast< IAcceptsLabels * >( net->getLastLayer() );
    if( lossLayer == 0 ) {
        throw runtime_error( "last layer of net should be a LossLayer class" );
    }
    lossLayer->calcGradInputFromLabels( labels );
    for( int layerIdx = numLayers - 2; layerIdx > 0; layerIdx-- ) {
        Layer *layer = net->getLayer( layerIdx );
        if( !layer->needsBackProp() ) {
            break;
        }
        layer->backward();
        if( layer->needsTrainerState() ) {
            updateWeights( layer->getWeightsWrapper(), layer->getGradWeightsWrapper(), 
                dynamic_cast< SGDState * >( layer->getTrainerState() ) );
            if( layer->biased() ) {
                updateWeights( layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< SGDState * >( layer->getBiasTrainerState() ) );
            }
        }
    }
}
VIRTUAL void SGD::bindState( NeuralNet *net ) {
    SGDStateMaker stateMaker;
    // go through network layers, and assign SGD objects (should probably rename these sometime somehow)
    for( int layerIdx = 0; layerIdx < net->getNumLayers(); layerIdx++ ) {
        Layer *layer = net->getLayer( layerIdx );
        if( layer->needsTrainerState() ) {
            TrainerState *state = layer->getTrainerState();
            SGDState *sgdState = dynamic_cast< SGDState *>( state );
            if( sgdState == 0 ) {
//                sgdState = new SGDState();
                layer->setTrainerState( &stateMaker );
            }
        }
    }
//    net->setTrainer( this );
}
STATIC SGD *SGD::instance( EasyCL *cl, float learningRate ) {
    SGD *sgd = new SGD( cl );
    sgd->setLearningRate( learningRate );
    return sgd;
}
STATIC SGD *SGD::instance( EasyCL *cl, float learningRate, float momentum ) {
    SGD *sgd = new SGD( cl );
    sgd->setLearningRate( learningRate );
    sgd->setMomentum( momentum );
    return sgd;
}
SGD::SGD( EasyCL *cl ) :
        Trainer( cl ),
        kernel( 0 ),
        momentum( 0.0f ) {
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

