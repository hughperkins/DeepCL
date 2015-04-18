// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include "NeuralNet.h"
#include "FullyConnectedLayer.h"
#include "ConvolutionalLayer.h"
#include "InputLayer.h"
#include "SoftMaxLayer.h"
#include "SquareLossLayer.h"
#include "CrossEntropyLoss.h"
#include "PoolingLayer.h"
#include "NormalizationLayer.h"
#include "RandomPatches.h"
#include "RandomTranslations.h"

#include "LayerMaker.h"

using namespace std;

Layer *SquareLossMaker::createLayer( Layer *previousLayer ) {
    return new SquareLossLayer( previousLayer, this );
}
Layer *CrossEntropyLossMaker::createLayer( Layer *previousLayer ) {
    return new CrossEntropyLoss( previousLayer, this );
}
Layer *SoftMaxMaker::createLayer( Layer *previousLayer ) {
    return new SoftMaxLayer( previousLayer, this );
}

