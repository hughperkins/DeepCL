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

#include "loaders/Loader.h"

#define VIRTUAL virtual
#define STATIC static

class ManifestLoaderv1 : public Loader {
public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL bool isFormatFor( char *header1024 );
    VIRTUAL void getDimensions( std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize );
    VIRTUAL void load( std::string filepath, unsigned char *data, int *labels, int startRecord, int numRecords );

    // [[[end]]]
};

