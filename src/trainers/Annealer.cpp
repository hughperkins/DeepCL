// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "trainers/Annealer.h"
#include "trainers/Trainer.h"
#include "EasyCL.h"
#include "util/stringhelper.h"
#include "net/NeuralNet.h"
#include "layer/Layer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

STATIC Annealer *Annealer::instance( EasyCL *cl, float learningRate, float anneal ) {
    Annealer *annealer = new Annealer( cl );
    annealer->setLearningRate( learningRate );
    annealer->setAnneal( anneal );
    return annealer;
}
Annealer::Annealer( EasyCL *cl ) :
    Trainer( cl ) {
    anneal = 1.0f;
//    epoch = -1;
}
VIRTUAL std::string Annealer::asString() {
    return "Annealer{ learningRate=" + toString( learningRate ) + ", anneal=" + 
        toString( anneal ) + " }";
}
VIRTUAL void Annealer::setAnneal( float anneal ) {
    this->anneal = anneal;
}
VIRTUAL void Annealer::updateWeights( CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper ) {
//    int numWeights = trainerState->numWeights;
//    // hmmmm, so all we need to do is calculate:
//    // annealedLearningRate = learningRate * pow( anneal, epoch )
//    // weightsWrapper = weightsWrapper - annealedLearningRate * gradWeightsWrapper
//    float annealedLearningRate = learningRate * pow( anneal, epoch );
//    CLWrapper *lastUpdateWrapper = trainerState->lastUpdateWrapper;
//    kernel  ->in( numWeights )
//            ->in( learningRate )
//            ->in( momentum )
//            ->inout( lastUpdateWrapper )
//            ->in( gradWeightsWrapper )
//            ->inout( weightsWrapper );
//    int globalSize = numWeights;
//    int workgroupSize = 64;
//    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
//    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
//    cl->finish();

//    if( weightDecay > 0 ) {
//        // apply weight decay, by multiplying the weights by (1.0f - weightDecay)
//        // so weightDecay == 0 means no decay; and weightDecay == 1.0f means
//        // weights go immediately to zero
//        multiplyInPlace->multiply( numWeights, 1.0f - weightDecay, weightsWrapper );
//    }
}
VIRTUAL BatchResult Annealer::train( NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput ) {
//    bindState( net );
//    net->forward( input );
//    int numLayers = net->getNumLayers();
//    LossLayer *lossLayer = dynamic_cast< LossLayer * >( net->getLastLayer() );
//    if( lossLayer == 0 ) {
//        throw runtime_error( "last layer of net should be a LossLayer class" );
//    }
//    lossLayer->calcGradInput( expectedOutput );
//    for( int layerIdx = numLayers - 2; layerIdx > 0; layerIdx-- ) {
//        Layer *layer = net->getLayer( layerIdx );
//        if( !layer->needsBackProp() ) {
//            break;
//        }
//        layer->backward();
//        if( layer->needsTrainerState() ) {
//            updateWeights( layer->getWeightsWrapper(), layer->getGradWeightsWrapper() );
//            if( layer->biased() ) {
//                updateWeights( layer->getBiasWrapper(), layer->getGradBiasWrapper() );
//            }
//        }
//    }
    return BatchResult(0,0);
}
VIRTUAL BatchResult Annealer::trainFromLabels( NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels ) {
//    bindState( net );
//    net->forward( input );
//    int numLayers = net->getNumLayers();
//    IAcceptsLabels *lossLayer = dynamic_cast< IAcceptsLabels * >( net->getLastLayer() );
//    if( lossLayer == 0 ) {
//        throw runtime_error( "last layer of net should be a LossLayer class" );
//    }
//    lossLayer->calcGradInputFromLabels( labels );
//    for( int layerIdx = numLayers - 2; layerIdx > 0; layerIdx-- ) {
//        Layer *layer = net->getLayer( layerIdx );
//        if( !layer->needsBackProp() ) {
//            break;
//        }
//        layer->backward();
//        if( layer->needsTrainerState() ) {
//            updateWeights( layer->getWeightsWrapper(), layer->getGradWeightsWrapper() );
//            if( layer->biased() ) {
//                updateWeights( layer->getBiasWrapper(), layer->getGradBiasWrapper() );
//            }
//        }
//    }
    return BatchResult(0,0);
}
VIRTUAL void Annealer::bindState( NeuralNet *net ) {
    // since we have no state, all we will do is strip any existing state,
    // so that if another trainer trains the net, it wont come across
    // some stale state
    for( int layerIdx = 0; layerIdx < net->getNumLayers(); layerIdx++ ) {
        Layer *layer = net->getLayer( layerIdx );
        if( layer->needsTrainerState() ) {
            TrainerState *state = layer->getTrainerState();
            if( state != 0 ) {
                layer->setTrainerState( 0 );
            }
        }
    }
}
//VIRTUAL bool Annealer::needEpoch() {
//    return true;
//}
//VIRTUAL void Annealer::setEpoch( int epoch ) {
//    this->epoch = epoch;
//}


