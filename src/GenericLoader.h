// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include <stdexcept>

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// not so generic, since only handles unsigned char for now :-P  But it's a start :-)
class DeepCL_EXPORT GenericLoader {
public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PUBLICAPI STATIC void getDimensions( std::string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize );
    PUBLICAPI STATIC void load( std::string imagesFilePath, float *images, int *labels, int startN, int numExamples );
    STATIC void load( std::string trainFilepath, unsigned char *images, int *labels );
    STATIC void load( std::string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples );

    // [[[end]]]
};

