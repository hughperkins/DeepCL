// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "NeuralNet.h"
#include "stringhelper.h"
#include "Trainer.h"
#include "MultiNet.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


Trainer::Trainer( EasyCL *cl ) :
    cl( cl ),
    learningRate( 0 ) {
}
VIRTUAL Trainer::~Trainer() {
}
VIRTUAL void Trainer::setLearningRate( float learningRate ) {
    this->learningRate = learningRate;
}
VIRTUAL std::string Trainer::asString() {
    return "Trainer{ learningRate=" + toString( learningRate ) + " }";
}
VIRTUAL void Trainer::train( Trainable *trainable, float const*input, float const*expectedOutput ) {
    MultiNet *multiNet = dynamic_cast< MultiNet *>( trainable );
    if( multiNet != 0 ) {
        for( int i = 0; i < multiNet->getNumNets(); i++ ) {
            Trainable *child = multiNet->getNet( i );
            this->train( child, input, expectedOutput );
        }
    } else {
        NeuralNet *net = dynamic_cast< NeuralNet * > ( trainable );
        this->train( net, input, expectedOutput );
    }
}
VIRTUAL void Trainer::trainFromLabels( Trainable *trainable, float const*input, int const*labels ) {
    MultiNet *multiNet = dynamic_cast< MultiNet *>( trainable );
    if( multiNet != 0 ) {
        for( int i = 0; i < multiNet->getNumNets(); i++ ) {
            Trainable *child = multiNet->getNet( i );
            this->trainFromLabels( child, input, labels );
        }
    } else {
        NeuralNet *net = dynamic_cast< NeuralNet * > ( trainable );
        this->trainFromLabels( net, input, labels );
    }
}


