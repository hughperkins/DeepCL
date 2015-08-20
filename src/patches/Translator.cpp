// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "Translator.h"

using namespace std;


void Translator::translate(int n, int numPlanes, int imageSize, int translateRows, int translateCols, float *source, float *destination) {
    const int cubeSize = numPlanes * imageSize * imageSize;
//    float *sourceCube = source + n * cubeSize;
    float *destinationCube = destination + n * cubeSize;
    memset(destinationCube, 0, sizeof(float) * cubeSize);
    const int rowCopyLength = imageSize - abs(translateCols);
    const int outColStart = translateCols > 0 ? translateCols : 0;
    const int inColStart = translateCols > 0 ? 0 : - translateCols;
    for(int plane = 0; plane < numPlanes; plane++) {
        float *upstreamImage = source + (n * numPlanes + plane) * imageSize * imageSize;
        float *outputImage = destination + (n * numPlanes + plane) * imageSize * imageSize;
        for(int inRow = 0; inRow < imageSize; inRow++) {
            const int outRow = inRow + translateRows;
            if(outRow < 0 || outRow > imageSize - 1) {
                continue;
            }
            memcpy(&(outputImage[ outRow * imageSize + outColStart ]), 
                &(upstreamImage[ inRow * imageSize + inColStart ]),
                rowCopyLength * sizeof(float));
        }        
    }
}


