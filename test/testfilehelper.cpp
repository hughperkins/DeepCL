#include <iostream>
using namespace std;

#include "gtest/gtest.h"

#include "FileHelper.h"
#include "test/myasserts.h"

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
        assertEquals( somefloats[i], newfloats[i], 0.0001 );
    }  
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
        assertEquals( somefloats[ 10000 + i], newfloats[i], 0.0001 );
    }  
}

