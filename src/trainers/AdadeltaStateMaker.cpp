// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "trainers/AdadeltaStateMaker.h"
#include "trainers/AdadeltaState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

TrainerState *AdadeltaStateMaker::instance(EasyCL *cl, int numWeights) {
    AdadeltaState *state = new AdadeltaState(cl, numWeights);
    return state;
}
VIRTUAL bool AdadeltaStateMaker::created(TrainerState *state) {
    return dynamic_cast< AdadeltaState * >(state) != 0;
}

