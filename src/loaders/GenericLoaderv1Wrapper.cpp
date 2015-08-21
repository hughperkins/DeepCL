// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "DeepCLDllExport.h"

#include "loaders/GenericLoader.h"
#include "loaders/GenericLoaderv1Wrapper.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLIC VIRTUAL std::string GenericLoaderv1Wrapper::getType() {
    return "GenericLoaderv1Wrapper";
}
PUBLIC VIRTUAL int GenericLoaderv1Wrapper::getN() {
    return N;
}
PUBLIC VIRTUAL int GenericLoaderv1Wrapper::getPlanes() {
    return planes;
}
PUBLIC VIRTUAL int GenericLoaderv1Wrapper::getImageSize() {
    return size;
}
PUBLIC GenericLoaderv1Wrapper::GenericLoaderv1Wrapper(std::string imagesFilepath) {
    this->imagesFilepath = imagesFilepath;
    GenericLoader::getDimensions(imagesFilepath.c_str(), &N, &planes, &size);
}
PUBLIC VIRTUAL int GenericLoaderv1Wrapper::getImageCubeSize() {
    return planes * size * size;
}
PUBLIC VIRTUAL void GenericLoaderv1Wrapper::load(unsigned char *data, int *labels, int startRecord, int numRecords) {
    GenericLoader::load(imagesFilepath.c_str(), data, labels, startRecord, numRecords);
}

