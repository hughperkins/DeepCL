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

using namespace std;

class NorbLoader {
public:
    // you need to delete[] this yourself, after use
    static unsigned char *loadImages( std::string filepath, int *p_N, int *p_numPlanes, int *p_boardSize ) {
        long imagesFilesize;
        char *imagesDataSigned = FileHelper::readBinary( filepath, &imagesFilesize );
        unsigned char *imagesDataUnsigned = reinterpret_cast< unsigned char *>(imagesDataSigned);
        unsigned int *imagesDataInt = reinterpret_cast< unsigned int *>( imagesDataSigned );
        int magic = imagesDataInt[0];
        std::cout << "magic: " << magic << std::endl;
        if( magic != 0x1e3d4c55 ) {
            throw std::runtime_error("magic value doesnt match expections: " + toString(magic) );
        }
        int ndim = imagesDataInt[1];
        int N = imagesDataInt[2];
        int numPlanes = imagesDataInt[3];
        int boardSize = imagesDataInt[4];
        int boardSizeRepeated = imagesDataInt[5];
        std::cout << "ndim " << ndim << " " << N << " " << numPlanes << " " << boardSize << " " << boardSizeRepeated << std::endl;
        checkSame( "boardSize", boardSize, boardSizeRepeated );

        int totalLinearSize = N * numPlanes * boardSize * boardSize;
        unsigned char*images = new unsigned char[ totalLinearSize ];
        memcpy( images, imagesDataUnsigned + 6 * 4, sizeof(unsigned char) * totalLinearSize );
        delete[] imagesDataUnsigned;

        *p_N = N;
        *p_numPlanes = numPlanes;
        *p_boardSize = boardSize;
        return images;
    }
    // you need to delete[] this yourself, after use
    static int *loadLabels( std::string filepath, int checkN ) {
        long imagesFilesize;
        char *imagesDataSigned = FileHelper::readBinary( filepath, &imagesFilesize );
        unsigned char *imagesDataUnsigned = reinterpret_cast< unsigned char *>(imagesDataSigned);
        unsigned int *imagesDataInt = reinterpret_cast< unsigned int *>( imagesDataSigned );
        int magic = imagesDataInt[0];
        std::cout << "magic: " << magic << std::endl;
        if( magic != 0x1e3d4c54 ) {
            throw std::runtime_error("magic value doesnt match expections: " + toString(magic) + " expected: " + toString( 0x1e3d4c54 ) );
        }
        int ndim = imagesDataInt[1];
        int N = imagesDataInt[2];
//        int d2 = imagesDataInt[3];
        checkSame( "ndim", 1, ndim );
        checkSame( "N", checkN, N );
//        checkSame( "d2", 1, d2 );
        
        int totalLinearSize = N;
        int *labels = new int[ N ];
        memcpy( labels, imagesDataInt + 5, sizeof(int) * totalLinearSize );
        delete[] imagesDataUnsigned;

        return labels;
    }
    static void writeImages( std::string filepath, unsigned char *images, int N, int numPlanes, int boardSize ) {
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
    static void writeLabels( std::string filepath, int *labels, int N ) {
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

protected:
    static void checkSame( std::string name, int one, int two ) {
        if( one != two ) {
            throw runtime_error( "Error, didnt match: " + name + " " + toString(one) + " != " + toString(two ) );
        }
    }
};

