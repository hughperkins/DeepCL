// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#define VIRTUAL virtual
#define STATIC static

class OpenCLHelper;

class PoolingBackprop {
public:
    OpenCLHelper *cl;

    const int numPlanes;
    const int inputBoardSize;
    const int poolingSize;

    const int outputBoardSize;

    inline int getInputIndex( int n, int plane, int row, int col ) {
        return ( ( n
            * numPlanes + plane )
            * inputBoardSize + row )
            * inputBoardSize + col;
    }
    inline int getResultIndex( int n, int plane, int row, int col ) {
        return ( ( n
            * numPlanes + plane )
            * outputBoardSize + row )
            * outputBoardSize + col;
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: PoolingBackprop
    // cppfile: PoolingBackprop.cpp

    STATIC PoolingBackprop *instance( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize );
    STATIC PoolingBackprop *instanceForTest( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize);
    STATIC PoolingBackprop *instanceSpecific( int idx, OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize );
    PoolingBackprop( OpenCLHelper *cl, int numPlanes, int inputBoardSize, int poolingSize );
    VIRTUAL void backpropErrors( int batchSize, float *errors, int *selectors, float *errorsForUpstream );
    VIRTUAL void backpropErrors( int batchSize, CLWrapper *errorsWrapper, CLWrapper *selectorsWrapper, CLWrapper *errorsForUpstreamWrapper );

    // [[[end]]]
};

