// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>

#include "ClConvolveDllExport.h"

class NeuralNet;

#define VIRTUAL virtual
#define STATIC static

class ClConvolve_EXPORT NetdefToNet {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC bool parseSubstring( NeuralNet *net, std::string substring, bool isLast );
    STATIC bool createNetFromNetdef( NeuralNet *net, std::string netdef );

    // [[[end]]]
};

