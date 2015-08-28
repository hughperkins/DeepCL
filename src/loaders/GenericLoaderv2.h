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

class Loader;

#define VIRTUAL virtual
#define STATIC static

// v1 loaders were stateless, all static functions
// but for imagenet manifest, we dont really want to load the manifest every single
// file read, so we make it stateful, hence GenericLoaderv2
class DeepCL_EXPORT GenericLoaderv2 {
    private:
    Loader *loader;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    GenericLoaderv2(std::string imagesFilepath);
    void load(float *images, int *labels, int startN, int numExamples);
    int getN();
    int getPlanes();
    int getImageSize();
    void load(unsigned char *images, int *labels);
    void load(unsigned char *images, int *labels, int startN, int numExamples);

    // [[[end]]]
};

