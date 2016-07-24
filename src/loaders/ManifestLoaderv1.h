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
    private:
    std::string imagesFilepath;
    int N;
    int planes;
    int size;

    bool hasLabels;
    std::string *files;
    int *labels;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    STATIC bool isFormatFor(std::string imagesFilepath);
    ManifestLoaderv1(std::string imagesFilepath);
    VIRTUAL std::string getType();
    VIRTUAL int getImageCubeSize();
    VIRTUAL int getN();
    VIRTUAL int getPlanes();
    VIRTUAL int getImageSize();
    VIRTUAL void load(unsigned char *data, int *labels, int startRecord, int numRecords);

    private:
    void init(std::string imagesFilepath);
    int readIntValue(std::vector< std::string > splitLine, std::string key);

    // [[[end]]]
};

