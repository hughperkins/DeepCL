// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "NorbLoader.h"
#include "FileHelper.h"
#include "Kgsv2Loader.h"
#include "StatefulTimer.h"
#include "MnistLoader.h"

#include "GenericLoader.h"

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI STATIC void GenericLoader::getDimensions( std::string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize ) {
    char *headerBytes = FileHelper::readBinaryChunk( trainFilepath, 0, 1024 );
    char type[1025];
    strncpy( type, headerBytes, 4 );
    type[4] = 0;
    unsigned int *headerInts = reinterpret_cast< unsigned int *>( headerBytes );
    if( string(type) == "mlv2" ) {
//        cout << "Loading as a Kgsv2 file" << endl;
        Kgsv2Loader::getDimensions( trainFilepath, p_numExamples, p_numPlanes, p_imageSize );
    } else if( headerInts[0] == 0x1e3d4c55 ) {
//        cout << "Loading as a Norb mat file" << endl;
        NorbLoader::getDimensions( trainFilepath, p_numExamples, p_numPlanes, p_imageSize );
    } else if( headerInts[0] == 0x03080000 ) {
        MnistLoader::getDimensions( trainFilepath, p_numExamples, p_numPlanes, p_imageSize );
    } else {
        cout << "headstring" << type << endl;
        throw runtime_error("Filetype of " + trainFilepath + " not recognised" );
    }
}

PUBLICAPI STATIC void GenericLoader::load( std::string imagesFilePath, float *images, int *labels, int startN, int numExamples ) {
//    cout << "GenericLoader::load " << numExamples << endl;
    int N, planes, size;
    getDimensions( imagesFilePath, &N, &planes, &size );
    unsigned char *ucImages = new unsigned char[ numExamples * planes * size * size ];
    load( imagesFilePath, ucImages, labels, startN, numExamples );
    int linearSize =  numExamples * planes * size * size;

    for( int i = 0; i < linearSize; i++ ) {
        images[i] = ucImages[i];
    }
    delete[] ucImages;
}

STATIC void GenericLoader::load( std::string trainFilepath, unsigned char *images, int *labels ) {
    load( trainFilepath, images, labels, 0, 0 );
}

STATIC void GenericLoader::load( std::string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples ) {
    StatefulTimer::timeCheck("GenericLoader::load start");
    char *headerBytes = FileHelper::readBinaryChunk( trainFilepath, 0, 1024 );
    char type[1025];
    strncpy( type, headerBytes, 4 );
    type[4] = 0;
    unsigned int *headerInts = reinterpret_cast< unsigned int *>( headerBytes );
    if( string(type) == "mlv2" ) {
//        cout << "Loading as a Kgsv2 file" << endl;
        Kgsv2Loader::load( trainFilepath, images, labels, startN, numExamples );
    } else if( headerInts[0] == 0x1e3d4c55 ) {
//        cout << "Loading as a Norb mat file" << endl;
        NorbLoader::load( trainFilepath, images, labels, startN, numExamples );
    } else if( headerInts[0] == 0x03080000 ) {
        MnistLoader::load( trainFilepath, images, labels, startN, numExamples );
    } else {
        cout << "headstring" << type << endl;
        throw runtime_error("Filetype of " + trainFilepath + " not recognised" );
    }
    StatefulTimer::timeCheck("GenericLoader::load end");
}


