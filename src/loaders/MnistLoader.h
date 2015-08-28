// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>

#include "util/FileHelper.h"
//#include "ImagesHelper.h"

#include "DeepCLDllExport.h"

#undef STATIC
#define STATIC static

class DeepCL_EXPORT MnistLoader {
public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC void getDimensions(std::string imagesFilePath,
    int *p_numExamples, int *p_numPlanes, int *p_imageSize);
    STATIC void load(std::string imagesFilePath, unsigned char *images, int *labels, int startN, int numExamples);
    STATIC int *loadLabels(std::string dir, std::string set, int *p_numImages);
    STATIC int readUInt(unsigned char *data, int location);
    STATIC void writeUInt(unsigned char *data, int location, int value);

    // [[[end]]]
};

