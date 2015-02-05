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

#include "DllImportExport.h"

#define VIRTUAL virtual
#define STATIC static

class ClConvolve_EXPORT NorbLoader {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC unsigned char *loadImages( std::string filepath, int *p_N, int *p_numPlanes, int *p_boardSize );
    STATIC unsigned char *loadImages( std::string filepath, int *p_N, int *p_numPlanes, int *p_boardSize, int maxN );
    STATIC int *loadLabels( std::string filepath, int Ntoget );
    STATIC void writeImages( std::string filepath, unsigned char *images, int N, int numPlanes, int boardSize );
    STATIC void writeLabels( std::string filepath, int *labels, int N );

    // [[[end]]]

protected:
    static void checkSame( std::string name, int one, int two ) {
        if( one != two ) {
            throw std::runtime_error( "Error, didnt match: " + name + " " + toString(one) + " != " + toString(two ) );
        }
    }
};

