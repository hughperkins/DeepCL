// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "trainers/NesterovStateMaker.h"
#include "trainers/NesterovState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

TrainerState *NesterovStateMaker::instance(EasyCL *cl, int numWeights) {
    NesterovState *sgd = new NesterovState(cl, numWeights);
    return sgd;
}
VIRTUAL bool NesterovStateMaker::created(TrainerState *state) {
    return dynamic_cast< NesterovState * >(state) != 0;
}

