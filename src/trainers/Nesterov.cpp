// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/stringhelper.h"
#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "loss/LossLayer.h"
#include "trainers/NesterovStateMaker.h"
#include "trainers/NesterovState.h"
#include "trainers/Nesterov.h"
#include "loss/IAcceptsLabels.h"
#include "clmath/MultiplyInPlace.h"
#include "batch/NetAction.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


VIRTUAL Nesterov::~Nesterov() {
    delete kernel;
    delete multiplyInPlace;
}
VIRTUAL void Nesterov::setMomentum( float momentum ) {
    this->momentum = momentum;
}
VIRTUAL void Nesterov::setWeightDecay( float weightDecay ) {
    this->weightDecay = weightDecay;
}
VIRTUAL std::string Nesterov::asString() {
    return "Nesterov{ learningRate=" + toString( learningRate ) + ", momentum=" + 
        toString( momentum ) + " }";
}
VIRTUAL void Nesterov::updateWeights( CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
        NesterovState *trainerState ) {
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

    if( weightDecay > 0 ) {
        // apply weight decay, by multiplying the weights by (1.0f - weightDecay)
        // so weightDecay == 0 means no decay; and weightDecay == 1.0f means
        // weights go immediately to zero
        multiplyInPlace->multiply( numWeights, 1.0f - weightDecay, weightsWrapper );
    }
}
VIRTUAL BatchResult Nesterov::train( NeuralNet *net, TrainingContext *context,
    float const*input, float const*expectedOutput ) {
    // learns one batch, including updating weights
    // doesnt have to think about running multiple batches,
    // or loading data, or anything like that
    bindState( net );
    net->forward( input );
    net->calcGrad();
    float loss = net->calcLoss( expectedOutput );

    int numLayers = net->getNumLayers();
//    LossLayer *lossLayer = dynamic_cast< LossLayer * >( net->getLastLayer() );
//    if( lossLayer == 0 ) {
//        throw runtime_error( "last layer of net should be a LossLayer class" );
//    }
//    lossLayer->calcGradInput( expectedOutput );
    for( int layerIdx = numLayers - 2; layerIdx > 0; layerIdx-- ) {
        Layer *layer = net->getLayer( layerIdx );
        if( !layer->needsBackProp() ) {
            break;
        }
//        layer->backward();
        if( layer->needsTrainerState() ) {
            updateWeights( layer->getWeightsWrapper(), layer->getGradWeightsWrapper(), 
                dynamic_cast< NesterovState * >( layer->getTrainerState() ) );
            if( layer->biased() ) {
                updateWeights( layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< NesterovState * >( layer->getBiasTrainerState() ) );
            }
        }
    }
    return BatchResult( loss, 0 );
}
VIRTUAL BatchResult Nesterov::trainFromLabels( NeuralNet *net, TrainingContext *context,
    float const*input, int const*labels ) {
//VIRTUAL void Nesterov::learn( float *input, float *expectedOutput ) { // learns one batch, including updating weights
                                  // doesnt have to think about running multiple batches,
                                  // or loading data, or anything like that
    // net->calcGrad();
//    cout << "Nesterov::train() istraining=" << net->isTraining << endl;
    bindState( net );
    net->forward( input );
    float loss = net->calcLossFromLabels( labels );
    int numRight = net->calcNumRight( labels );
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
                dynamic_cast< NesterovState * >( layer->getTrainerState() ) );
            if( layer->biased() ) {
                updateWeights( layer->getBiasWrapper(), layer->getGradBiasWrapper(),
                    dynamic_cast< NesterovState * >( layer->getBiasTrainerState() ) );
            }
        }
    }
    return BatchResult( loss, numRight );
}
VIRTUAL void Nesterov::bindState( NeuralNet *net ) {
    NesterovStateMaker stateMaker;
    // go through network layers, and assign Nesterov objects (should probably rename these sometime somehow)
    for( int layerIdx = 0; layerIdx < net->getNumLayers(); layerIdx++ ) {
        Layer *layer = net->getLayer( layerIdx );
        if( layer->needsTrainerState() ) {
            TrainerState *state = layer->getTrainerState();
            NesterovState *sgdState = dynamic_cast< NesterovState *>( state );
            if( sgdState == 0 ) {
//                sgdState = new NesterovState();
                layer->setTrainerState( &stateMaker );
            }
        }
    }
//    net->setTrainer( this );
}
STATIC Nesterov *Nesterov::instance( EasyCL *cl, float learningRate ) {
    Nesterov *sgd = new Nesterov( cl );
    sgd->setLearningRate( learningRate );
    return sgd;
}
STATIC Nesterov *Nesterov::instance( EasyCL *cl, float learningRate, float momentum ) {
    Nesterov *sgd = new Nesterov( cl );
    sgd->setLearningRate( learningRate );
    sgd->setMomentum( momentum );
    return sgd;
}
Nesterov::Nesterov( EasyCL *cl ) :
        Trainer( cl ),
        kernel( 0 ),
        momentum( 0.0f ),
        weightDecay( 0.0f ) {
    string options = "";
    multiplyInPlace = new MultiplyInPlace( cl );
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

