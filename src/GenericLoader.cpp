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

#include "GenericLoader.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

STATIC void GenericLoader::getDimensions( std::string trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize ) {
    char *headerBytes = FileHelper::readBinaryChunk( trainFilepath, 0, 1024 );
    char type[1025];
    strncpy( type, headerBytes, 4 );
    type[4] = 0;
    unsigned int *headerInts = reinterpret_cast< unsigned int *>( headerBytes );
    if( string(type) == "mlv2" ) {
//        cout << "Loading as a Kgsv2 file" << endl;
        return Kgsv2Loader::getDimensions( trainFilepath, p_numExamples, p_numPlanes, p_imageSize );
    } else if( headerInts[0] == 0x1e3d4c55 ) {
//        cout << "Loading as a Norb mat file" << endl;
        return NorbLoader::getDimensions( trainFilepath, p_numExamples, p_numPlanes, p_imageSize );
    } else {
        cout << "headstring" << type << endl;
        throw runtime_error("Filetype of " + trainFilepath + " not recognised" );
    }
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
    } else {
        cout << "headstring" << type << endl;
        throw runtime_error("Filetype of " + trainFilepath + " not recognised" );
    }
    StatefulTimer::timeCheck("GenericLoader::load end");
}


