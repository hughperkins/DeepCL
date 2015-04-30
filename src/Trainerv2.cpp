// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "OpenCLHelper.h"
#include "NeuralNet.h"
#include "stringhelper.h"
#include "Trainerv2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


Trainerv2::Trainerv2( OpenCLHelper *cl, NeuralNet *net ) :
    cl( cl ),
    net( net ),
    learningRate( 0 ) {
}
VIRTUAL Trainerv2::~Trainerv2() {
}
VIRTUAL void Trainerv2::setLearningRate( float learningRate ) {
    this->learningRate = learningRate;
}
VIRTUAL std::string Trainerv2::asString() {
    return "Trainerv2{ learningRate=" + toString( learningRate ) + " }";
}

