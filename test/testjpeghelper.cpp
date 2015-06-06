// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
using namespace std;

#include "util/JpegHelper.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include "DeepCLDllExport.h" // contains uchar typedef

TEST( testjpeghelper, writeread ) {
    // write a jpeg, read it, check same
    int planes = 1;
    int imageSize = 28;

    uchar *data = new uchar[imageSize * imageSize];
    for( int row = 0; row < imageSize; row++ ) {
        for( int col = 0; col < imageSize; col++ ) {
            data[ row * imageSize + col ] = (uchar)( ( 10 + row * 5 - col * 12 ) % 255 );
        }
    }
    JpegHelper::write("~foo.jpeg", 1, 28, 28, data );
    uchar *data2 = new uchar[imageSize * imageSize];
    JpegHelper::read( "~foo.jpeg", 1, 28, 28, data2 );
    int linearSize = planes * imageSize * imageSize;
    bool allOk = true;
    for( int i = 0; i < linearSize; i++ ) {
        int diff = data[i] - data2[i];
        int absdiff = diff > 0 ? diff : - diff;
        if( absdiff > 50 ) {
            allOk = false;
            cout << "diff [" << i << "]: " << (int)data[i] << " " << (int)data2[i] << endl;
        }
    }
    EXPECT_TRUE( allOk );

    delete[] data2;
    delete[] data;
}

