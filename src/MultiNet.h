// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

#define VIRTUAL virtual
#define STATIC static

#include "DllImportExport.h"

class NeuralNet;

// This handles grouping several NeuralNets into one single MultiNet
class ClConvolve_EXPORT MultiNet {
    std::vector<NeuralNet * > nets;

public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    MultiNet( int numNets, NeuralNet *model );

    // [[[end]]]
};

