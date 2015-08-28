// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "trainers/AdagradStateMaker.h"
#include "trainers/AdagradState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

AdagradStateMaker::AdagradStateMaker(float fudgeFactor) {
    this->fudgeFactor = fudgeFactor;
}
TrainerState *AdagradStateMaker::instance(EasyCL *cl, int numWeights) {
    AdagradState *state = new AdagradState(cl, numWeights, fudgeFactor);
    return state;
}
VIRTUAL bool AdagradStateMaker::created(TrainerState *state) {
    return dynamic_cast< AdagradState * >(state) != 0;
}

