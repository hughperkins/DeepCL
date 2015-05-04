// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


// converts cifar bin format files to norb .mat format files

#include <stdexcept>
#include <iostream>

#include "test/CifarLoader.h"
#include "util/stringhelper.h"
#include "loaders/NorbLoader.h"

using namespace std;

void doTestFiles( string dir ) {
    string cifarFilename = dir + "/test_batch.bin";

    string matDatFilename = dir + "/test-dat.mat";
    string matCatFilename = dir + "/test-cat.mat";

    int N = CifarLoader::getNumExamples( cifarFilename );
    int imagesSize = CifarLoader::getImagesSize( cifarFilename );
    cout << "num examples: " << N << " imagesSize " << imagesSize << endl;
    unsigned char *images = new unsigned char[imagesSize];
    int *labels = new int[ N ];
    CifarLoader::load( cifarFilename, images, labels );
    
    const int imageSize = 32;
    const int numPlanes = 3;
    NorbLoader::writeLabels( matCatFilename, labels, N );
    NorbLoader::writeImages( matDatFilename, images, N, numPlanes, imageSize );

    delete[] images;
    delete[] labels;
}

void doTrainingFiles( string dir ) {
    const int N = 50000; // yeah, we just hard-code this...
    const int numPlanes = 3;
    const int imageSize = 32;
    int *labels = new int[N];
    const int cubeSize = numPlanes * imageSize * imageSize;
    unsigned char *images = new unsigned char[ N * cubeSize ];
    for( int batch = 1; batch <= 5; batch++ ) {
        string cifarFilename = dir + "/data_batch_" + toString( batch ) + ".bin";
        CifarLoader::load( cifarFilename, images + ( batch - 1 ) * 10000 * cubeSize, labels + ( batch - 1 ) * 10000 );
    }

    string matDatFilename = dir + "/train-dat.mat";
    string matCatFilename = dir + "/train-cat.mat";

    NorbLoader::writeLabels( matCatFilename, labels, N );
    NorbLoader::writeImages( matDatFilename, images, N, numPlanes, imageSize );

    delete[] images;
    delete[] labels;
}

void go( string dir ) {
    // string matDir = argv[2];

    doTrainingFiles( dir );
    doTestFiles( dir );
}

// this is for cifar10 apparently...
int main( int argc, char *argv[] ) {
    if( argc != 2  ) {
        cout << "Usage: " << argv[0] << " [directory path]" << endl;
        return -1;
    }

    string dir = argv[1];
//    string setName = argv[2];

    try {
        go( dir );
    } catch( runtime_error e ) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }

    return 0;
}


