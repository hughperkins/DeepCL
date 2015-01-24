// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "png++/png.hpp"

#include "test/NorbLoader.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

using namespace std;

TEST( SLOW_testNorbLoader, basic ) {
    int N;
    int numPlanes;
    int boardSize;
    unsigned char *images = NorbLoader::loadTrainingImages( "../data/norb", &N, &numPlanes, &boardSize );
    png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >( boardSize * 2, boardSize * 2 );
    for( int imageRow = 0; imageRow < 2; imageRow++ ) {
        for( int imageCol = 0; imageCol < 2; imageCol++ ) {
            for( int i = 0; i < boardSize; i++ ) {
                for( int j = 0; j < boardSize; j++ ) {
                   int value = images[(imageRow*2+imageCol) * boardSize * boardSize + i*boardSize + j];
                   (*image)[i + imageRow*boardSize][j + imageCol*boardSize] = png::rgb_pixel( value, value, value );
                }
            }
        }
    }
    remove( "testNorbLoader.png" );
    image->write( "testNorbLoader.png" );
    delete[] images;
}

