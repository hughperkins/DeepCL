// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "trainers/RmspropStateMaker.h"
#include "trainers/RmspropState.h"

using namespace std;
using namespace easycl;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

TrainerState *RmspropStateMaker::instance(EasyCL *cl, int numWeights) {
    RmspropState *state = new RmspropState(cl, numWeights);
    return state;
}
VIRTUAL bool RmspropStateMaker::created(TrainerState *state) {
    return dynamic_cast< RmspropState * >(state) != 0;
}

