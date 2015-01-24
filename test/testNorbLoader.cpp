// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#ifdef PNG_AVAILABLE
#include "png++/png.hpp"
#endif //PNG_AVAILABLE

#include "test/NorbLoader.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

using namespace std;

TEST( SLOW_testNorbLoader, basic ) {
    int N;
    int numPlanes;
    int boardSize;
    string norbDataDir = "../data/norb";
    unsigned char *images = NorbLoader::loadImages( norbDataDir + "/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat", &N, &numPlanes, &boardSize );
    int *labels = NorbLoader::loadLabels( norbDataDir + "/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat", N );
    cout << "labels here, please open testNorbLoader.png, and compare" << endl;
    for( int i = 0; i < 4; i++ ) {
        string thisRow = "";
        for( int j = 0; j < 4; j++ ) {
            thisRow += toString( labels[i*4+j] ) + " ";
        }
        cout << thisRow << endl;
    }
#ifdef PNG_AVAILABLE
    png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >( boardSize * 8, boardSize * 4 );
    for( int imageRow = 0; imageRow < 4; imageRow++ ) {
        for( int imageCol = 0; imageCol < 4; imageCol++ ) {
            for( int p = 0; p < 2; p++ ) {
                for( int i = 0; i < boardSize; i++ ) {
                    for( int j = 0; j < boardSize; j++ ) {
                           int value = images[((imageRow*4+imageCol )*2 + p) * boardSize * boardSize + i*boardSize + j];
                       (*image)[i + imageRow*boardSize][j + (imageCol*2+p)*boardSize] = png::rgb_pixel( value, value, value );
                    }
                }
            }
        }
    }
    FileHelper::remove( "testNorbLoader.png" );
    image->write( "testNorbLoader.png" );
#endif
    delete[] images;
}

