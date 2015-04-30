// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "stringhelper.h"
#include "Trainer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


Trainer::Trainer( OpenCLHelper *cl, NeuralNet *net ) :
    cl( cl ),
    net( net ),
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

