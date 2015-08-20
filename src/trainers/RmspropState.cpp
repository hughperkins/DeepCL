// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "trainers/RmspropState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL RmspropState::~RmspropState() {
    delete meanSquareWrapper;
    delete[] meanSquare;
}

RmspropState::RmspropState(EasyCL *cl, int numWeights) :
        numWeights(numWeights) {
    meanSquare = new float[numWeights];
    for(int i = 0; i < numWeights; i++) {
        meanSquare[i] = 0.0000001f; // should move this into fudgefactor I guess?
    }
    meanSquareWrapper = cl->wrap(numWeights, meanSquare);
    meanSquareWrapper->copyToDevice();
}


