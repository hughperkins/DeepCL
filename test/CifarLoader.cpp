// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <cstring>

#include "util/FileHelper.h"

#include "CifarLoader.h"

using namespace std;

int CifarLoader::getNumExamples( std::string filepath ) {
    long fileSize = FileHelper::getFilesize( filepath );
    return fileSize / ( 32 * 32 * 3 + 1 );
}
int CifarLoader::getImagesSize( std::string filepath ) { // you can use this to help allocate the images array
    long fileSize = FileHelper::getFilesize( filepath );
    return fileSize / ( 32 * 32 * 3 + 1 ) * ( 32 * 32 * 3 );
}
void CifarLoader::load( std::string filepath, unsigned char *images, int *labels ) { // you need to pre-allocate these arrays
    int numExamples = getNumExamples( filepath );
    long fileSize;
    char *bytes = FileHelper::readBinary( filepath, &fileSize );
    unsigned char *ubytes = reinterpret_cast< unsigned char * >( bytes );
    for( int n = 0; n < numExamples; n++ ) {
        unsigned char *labelimage = ubytes + n * ( 32 * 32 * 3 + 1 );
        unsigned char *p_label = labelimage;
        unsigned char *image = labelimage + 1;
        labels[n] = *p_label;
        unsigned char *targetCube = images + n * ( 32 * 32 * 3 );
        memcpy( targetCube, image, 32 * 32 * 3 );
    }
}

