// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include "Layer.h"

#include "NeuralNet.h"

#include "NeuralNetMould.h"

using namespace std;

NeuralNet *NeuralNetMould::instance() {
    if( _numPlanes == 0 ) {
        throw runtime_error("Must provide ->planes(planes)");
    }
    if( _boardSize == 0 ) {
        throw runtime_error("Must provide ->boardSize(boardSize)");
    }
    NeuralNet *net = new NeuralNet( _numPlanes, _boardSize );
    delete this;
    return net;
}

