// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "util/FileHelper.h"
#include "util/StatefulTimer.h"
#include "loaders/Loader.h"
#include "loaders/GenericLoaderv1Wrapper.h"
#include "loaders/GenericLoaderv2.h"

#ifdef LIBJPEG_FOUND
#include "loaders/ManifestLoaderv1.h"
#endif

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLIC GenericLoaderv2::GenericLoaderv2(std::string imagesFilepath) {
    loader = 0;
    #ifdef LIBJPEG_FOUND
    if(ManifestLoaderv1::isFormatFor(imagesFilepath) ) {
        loader = new ManifestLoaderv1(imagesFilepath);
    }
    #endif
    if(loader == 0) {
        loader = new GenericLoaderv1Wrapper(imagesFilepath);
    }
}

PUBLIC void GenericLoaderv2::load(float *images, int *labels, int startN, int numExamples) {
    int linearSize =  numExamples * loader->getImageCubeSize();
    unsigned char *ucImages = new unsigned char[ linearSize ];

    load(ucImages, labels, startN, numExamples);

    for(int i = 0; i < linearSize; i++) {
        images[i] = ucImages[i];
    }
    delete[] ucImages;
}
PUBLIC int GenericLoaderv2::getN() {
    return loader->getN();
}
PUBLIC int GenericLoaderv2::getPlanes() {
    return loader->getPlanes();
}
PUBLIC int GenericLoaderv2::getImageSize() {
    return loader->getImageSize();
}
PUBLIC void GenericLoaderv2::load(unsigned char *images, int *labels) {
    load(images, labels, 0, 0);
}

PUBLIC void GenericLoaderv2::load(unsigned char *images, int *labels, int startN, int numExamples) {
    StatefulTimer::timeCheck("GenericLoaderv2::load start");

    loader->load(images, labels, startN, numExamples);

    StatefulTimer::timeCheck("GenericLoaderv2::load end");
}


