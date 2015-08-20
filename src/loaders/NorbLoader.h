// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <iostream>
#include <cstring>

#include "util/stringhelper.h"
#include "util/FileHelper.h"

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class DeepCL_EXPORT NorbLoader {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC void getDimensions(std::string trainFilepath, int *p_N, int *p_numPlanes, int *p_imageSize);
    STATIC void load(std::string trainFilepath, unsigned char *images, int *labels);
    STATIC void load(std::string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples);
    STATIC int *loadLabels(std::string labelsfilepath, int numExamples);
    STATIC unsigned char *loadImages(std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize);
    STATIC unsigned char *loadImages(std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize, int numExamples);
    STATIC unsigned char *loadImages(std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize, int startN, int numExamples);
    STATIC void loadImages(unsigned char *images, std::string filepath, int *p_N, int *p_numPlanes, int *p_imageSize, int startN, int numExamples);
    STATIC void loadLabels(int *labels, std::string filepath, int startN, int numExamples);
    STATIC void writeImages(std::string filepath, unsigned char *images, int N, int numPlanes, int imageSize);
    STATIC void writeLabels(std::string filepath, int *labels, int N);

    // [[[end]]]

protected:
    static void checkSame(std::string name, int one, int two) {
        if(one != two) {
            throw std::runtime_error("Error, didnt match: " + name + " " + toString(one) + " != " + toString(two) );
        }
    }
};

