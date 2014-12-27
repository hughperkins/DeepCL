// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "LayerMaker.h"

#include "NeuralNet.h"

Layer *FullyConnectedMaker::insert() {
    Layer *layer = net->addFullyConnected( _numPlanes, _boardSize );
    delete this;
    return layer;
}

Layer *ConvolutionalMaker::insert() {
    Layer *layer = net->addConvolutional( _numFilters, _filterSize );
    delete this;
    return layer;
}


