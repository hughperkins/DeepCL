// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
using namespace std;

#include "util/FileHelper.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"


TEST( testfilehelper, testfilehelper ) {
    int N = 100000;
    float *somefloats = new float[N];
    for( int i = 0; i < N; i++ ) {
        somefloats[i] = i * 5.0 / 3.0;
    }
    FileHelper::writeBinary("foo.dat", reinterpret_cast<char*>(somefloats), N * sizeof(float) );
    float *newfloats = new float[N];
    long bytesread = 0;
    char *dataread = FileHelper::readBinary("foo.dat", &bytesread );
    for( int i = 0; i < N; i++ ) {
        newfloats[i] = reinterpret_cast<float*>(dataread)[i];
    }
    delete[] dataread;
    for( int i = 0; i < N; i++ ) {
        EXPECT_FLOAT_NEAR( somefloats[i], newfloats[i] );
    }  
    EXPECT_EQ( N * sizeof(float), FileHelper::getFilesize( "foo.dat" ) );
}

TEST( testfilehelper, testreadchunk ) {
    int N = 100000;
    float *somefloats = new float[N];
    for( int i = 0; i < N; i++ ) {
        somefloats[i] = i * 5.0 / 3.0;
    }
    FileHelper::writeBinary("foo.dat", reinterpret_cast<char*>(somefloats), N * sizeof(float) );
    float *newfloats = new float[100];
    char *dataread = FileHelper::readBinaryChunk("foo.dat", 10000 * sizeof(float), 100 * sizeof(float) );
    for( int i = 0; i < 100; i++ ) {
        newfloats[i] = reinterpret_cast<float*>(dataread)[i];
    }
    delete[] dataread;
    for( int i = 0; i < 100; i++ ) {
        EXPECT_FLOAT_NEAR( somefloats[ 10000 + i], newfloats[i] );
    }  
}

