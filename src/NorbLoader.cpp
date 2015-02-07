// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <iostream>
#include <cstring>

#include "stringhelper.h"
#include "FileHelper.h"

#include "NorbLoader.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

STATIC void NorbLoader::getDimensions( std::string trainFilepath, int *p_N, int *p_numPlanes, int *p_boardSize, int *p_imagesLinearSize ) {
    char*headerBytes = FileHelper::readBinaryChunk( trainFilepath, 0, 6 * 4 );
    unsigned int *headerValues = reinterpret_cast< unsigned int *>( headerBytes );

    int magic = headerValues[0];
//    std::cout << "magic: " << magic << std::endl;
    if( magic != 0x1e3d4c55 ) {
        throw std::runtime_error("magic value doesnt match expections: " + toString(magic) );
    }
    int ndim = headerValues[1];
    int N = headerValues[2];
    int numPlanes = headerValues[3];
    int boardSize = headerValues[4];
    int boardSizeRepeated = headerValues[5];
//    std::cout << "ndim " << ndim << " " << N << " " << numPlanes << " " << boardSize << " " << boardSizeRepeated << std::endl;
    checkSame( "boardSize", boardSize, boardSizeRepeated );

//    if( maxN > 0 ) {
//        N = min( maxN, N );
//    }
    int totalLinearSize = N * numPlanes * boardSize * boardSize;
    *p_N = N;
    *p_numPlanes = numPlanes;
    *p_boardSize = boardSize;
    *p_imagesLinearSize = totalLinearSize;
}

STATIC void NorbLoader::load( std::string trainFilepath, unsigned char *images, int *labels, int startN, int numExamples ) {
    int N, numPlanes, boardSize;
    // I know, this could be optimized a bit, to remove the intermediate arrays...
    unsigned char *loadedImages = loadImages( trainFilepath, &N, &numPlanes, &boardSize, startN + numExamples );
//    int totalLinearSize = numExamples  * numPlanes * boardSize * boardSize;
    memcpy( images, loadedImages + startN * numPlanes * boardSize * boardSize, numExamples * numPlanes * boardSize * boardSize * sizeof( unsigned char ) );
    int *loadedLabels = loadLabels( replace( trainFilepath, "-dat.mat","-cat.mat"), startN + numExamples );
    memcpy( labels, loadedLabels + startN, sizeof( int ) * numExamples );
    delete []loadedImages;
    delete[] loadedLabels;
}

STATIC unsigned char *NorbLoader::loadImages( std::string filepath, int *p_N, int *p_numPlanes, int *p_boardSize ) {
    return loadImages( filepath, p_N, p_numPlanes, p_boardSize, 0 );
}

// you need to delete[] this yourself, after use
STATIC unsigned char *NorbLoader::loadImages( std::string filepath, int *p_N, int *p_numPlanes, int *p_boardSize, int maxN ) {
    char*headerBytes = FileHelper::readBinaryChunk( filepath, 0, 6 * 4 );
    unsigned int *headerValues = reinterpret_cast< unsigned int *>( headerBytes );

    int magic = headerValues[0];
//    std::cout << "magic: " << magic << std::endl;
    if( magic != 0x1e3d4c55 ) {
        throw std::runtime_error("magic value doesnt match expections: " + toString(magic) );
    }
    int ndim = headerValues[1];
    int N = headerValues[2];
    int numPlanes = headerValues[3];
    int boardSize = headerValues[4];
    int boardSizeRepeated = headerValues[5];
//    std::cout << "ndim " << ndim << " " << N << " " << numPlanes << " " << boardSize << " " << boardSizeRepeated << std::endl;
    checkSame( "boardSize", boardSize, boardSizeRepeated );

    if( maxN > 0 ) {
        N = min( maxN, N );
    }
    int totalLinearSize = N * numPlanes * boardSize * boardSize;
    char *imagesDataSigned = FileHelper::readBinaryChunk( filepath, 6 * 4, totalLinearSize );
    unsigned char *imagesDataUnsigned = reinterpret_cast< unsigned char *>(imagesDataSigned);

    *p_N = N;
    *p_numPlanes = numPlanes;
    *p_boardSize = boardSize;
    return imagesDataUnsigned;
}
// you need to delete[] this yourself, after use
STATIC int *NorbLoader::loadLabels( std::string filepath, int Ntoget ) {
    char*headerBytes = FileHelper::readBinaryChunk( filepath, 0, 6 * 5 );
    unsigned int *headerValues = reinterpret_cast< unsigned int *>( headerBytes );

    int magic = headerValues[0];
//    std::cout << "magic: " << magic << std::endl;
    if( magic != 0x1e3d4c54 ) {
        throw std::runtime_error("magic value doesnt match expections: " + toString(magic) + " expected: " + toString( 0x1e3d4c54 ) );
    }
    int ndim = headerValues[1];
    int N = headerValues[2];
//    checkSame( "ndim", 1, ndim );
    if( Ntoget > N ) {
        throw runtime_error("error: you asked for " + toString( Ntoget ) + " labels, but only found: " + toString( N ) );
    }
    N = Ntoget;

    int totalLinearSize = N;
    char *labelsAsByteArray = FileHelper::readBinaryChunk( filepath, 5 * 4, totalLinearSize * 4 );
    int *labels = reinterpret_cast< int *>(labelsAsByteArray);
    return labels;
}
STATIC void NorbLoader::writeImages( std::string filepath, unsigned char *images, int N, int numPlanes, int boardSize ) {
    int totalLinearSize = N * numPlanes * boardSize * boardSize;

    long imagesFilesize = totalLinearSize + 6 * 4; // magic, plus num dimensions, plus 4 dimensions
    char *imagesDataSigned = new char[ imagesFilesize ];
    unsigned int *imagesDataInt = reinterpret_cast< unsigned int *>( imagesDataSigned );
    unsigned char *imagesDataUnsigned = reinterpret_cast< unsigned char *>(imagesDataSigned);
    imagesDataInt[0] = 0x1e3d4c55;
    imagesDataInt[1] = 4;
    imagesDataInt[2] = N;
    imagesDataInt[3] = numPlanes;
    imagesDataInt[4] = boardSize;
    imagesDataInt[5] = boardSize;
    memcpy( imagesDataUnsigned + 6 * sizeof(int), images, totalLinearSize * sizeof( unsigned char ) );
    FileHelper::writeBinary( filepath, imagesDataSigned, imagesFilesize );
}
STATIC void NorbLoader::writeLabels( std::string filepath, int *labels, int N ) {
    int totalLinearSize = N;

    long imagesFilesize = totalLinearSize * 4 + 5 * 4; // magic, plus num dimensions, plus 3 dimensions
    char *imagesDataSigned = new char[ imagesFilesize ];
    unsigned int *imagesDataInt = reinterpret_cast< unsigned int *>( imagesDataSigned );
    unsigned char *imagesDataUnsigned = reinterpret_cast< unsigned char *>(imagesDataSigned);
    imagesDataInt[0] = 0x1e3d4c54;
    imagesDataInt[1] = 1;
    imagesDataInt[2] = N;
    imagesDataInt[3] = 1;
    imagesDataInt[4] = 1;
    memcpy( imagesDataUnsigned + 5 * sizeof(int), labels, totalLinearSize * sizeof( int ) );
    FileHelper::writeBinary( filepath, imagesDataSigned, imagesFilesize );
}

