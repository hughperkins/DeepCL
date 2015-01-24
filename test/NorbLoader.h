// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <iostream>

#include "stringhelper.h"
#include "FileHelper.h"

using namespace std;

class NorbLoader {
public:
    // you need to delete[] this yourself, after use
    static unsigned char *loadTrainingImages( std::string dirpath, int *p_N, int *p_numPlanes, int *p_boardSize ) {
        string filepath = dirpath + "/" + "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat";
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

        int totalLinearSize = N * numPlanes * boardSize * boardSize;
//        float *images = new float[ totalLinearSize ];
//        for( int i = 0; i < totalLinearSize; i++ ) {
//            images[i] = imagesDataUnsigned[i];
//        }

        *p_N = N;
        *p_numPlanes = numPlanes;
        *p_boardSize = boardSize;
        return imagesDataUnsigned;
//        return images;
    }
    // you need to delete[] this yourself, after use
    static int *loadTrainingLabels( std::string dirpath ) {
        std::string filepath = dirpath + "/" + "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat";
    }
};

