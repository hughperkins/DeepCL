// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

#define VIRTUAL virtual
#define STATIC static

// passed to trainers, for each training batch
class TrainingContext {
public:
    const int epoch; // zero-based
    // could add in other things here, like batch number etc...

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    TrainingContext( int epoch );
    int getEpoch();

    // [[[end]]]
};

