// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "util/StatefulTimer.h"
#include "trainers/SGDState.h"

using namespace std;
using namespace easycl;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL SGDState::~SGDState() {
    delete lastUpdateWrapper;
    delete[] lastUpdate;
}

SGDState::SGDState(EasyCL *cl, int numWeights) :
        numWeights(numWeights)
    { // should we handle bias separately?  maybe... not?
      // or each layer could have one trainer for biases, and one for the
      // non-biases?  Maybe kind of ok?

    // lastUpdate buffer never needs to change size,
    //  since number of weights is invariant with batchSize etc
    lastUpdate = new float[numWeights];
    for(int i = 0; i < numWeights; i++) {
        lastUpdate[i] = 0.0f;
    }
    lastUpdateWrapper = cl->wrap(numWeights, lastUpdate);
    lastUpdateWrapper->copyToDevice();
}

