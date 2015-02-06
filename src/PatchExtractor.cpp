// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <cstring>

#include "PatchExtractor.h"

using namespace std;

void PatchExtractor::extractPatch( int n, int numPlanes, int boardSize, int patchSize, int patchRow, int patchCol, float *source, float *destination ) {
//    int n = 0;
    for( int plane = 0; plane < numPlanes; plane++ ) {
        float *upstreamBoard = source + ( n * numPlanes + plane ) * boardSize * boardSize;
        float *outputBoard = destination + ( n * numPlanes + plane ) * patchSize * patchSize;
        for( int outRow = 0; outRow < patchSize; outRow++ ) {
            const int inRow = outRow + patchRow;
            memcpy( &(outputBoard[ outRow * patchSize ]), 
                &(upstreamBoard[ inRow * boardSize + patchCol ]),
                patchSize * sizeof(float) );
        }        
    }
}


