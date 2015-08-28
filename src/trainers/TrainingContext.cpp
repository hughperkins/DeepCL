// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "trainers/TrainingContext.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

TrainingContext::TrainingContext(int epoch, int batch) :
        epoch(epoch),
        batch(batch) {
}
int TrainingContext::getEpoch() {
    return epoch;
}
int TrainingContext::getBatch() {
    return batch;
}

