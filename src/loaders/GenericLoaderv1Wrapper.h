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

// wraps GenericLoader v1, so behaves like a GenericLoaderv2 loader
// v1 loaders were stateless, all static functions
// but for imagenet manifest, we dont really want to load the manifest every single
// file read, so we make it stateful, hence GenericLoaderv2
class GenericLoaderv1Wrapper : public Loader {
    private:
    std::string imagesFilepath;
    int N;
    int planes;
    int size;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    VIRTUAL std::string getType();
    VIRTUAL int getN();
    VIRTUAL int getPlanes();
    VIRTUAL int getImageSize();
    GenericLoaderv1Wrapper(std::string imagesFilepath);
    VIRTUAL int getImageCubeSize();
    VIRTUAL void load(unsigned char *data, int *labels, int startRecord, int numRecords);

    // [[[end]]]
};

