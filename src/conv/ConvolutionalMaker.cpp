// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "conv/ConvolutionalLayer.h"

#include "conv/ConvolutionalMaker.h"

using namespace std;

Layer *ConvolutionalMaker::createLayer(Layer *previousLayer) {
    if(_numFilters == 0) {
        throw runtime_error("Must provide ->numFilters(numFilters)");
    }
    if(_filterSize == 0) {
        throw runtime_error("Must provide ->filterSize(filterSize)");
    }
    Layer *layer = new ConvolutionalLayer(cl, previousLayer, this);
    return layer;
}

