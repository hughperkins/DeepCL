// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


// read from the provided norb files, and :
// - create randomized version, for training
// - create smaller version, randomly sampled from the provided testing, for testing
//   (since, if only need testing accuracy accurate to 0.1%, only need 1000 testing samples.... )

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <random>

#include "test/NorbLoader.h"

using namespace std;

int main( int argc, char *argv[] ) {
    string norbDir = "../data/norb";

    int N, numPlanes, boardSize, boardSizeRepeated;
    unsigned char *training = NorbLoader::loadImages( norbDir + "/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat", &N, &numPlanes, &boardSize );
    // create random sequence of examples
    vector<int> sequence;
    sequence.reserve(N);
    for( int n = 0; n < N; n++ ) {
        sequence.push_back( n );
    }
    shuffle( sequence.begin(), sequence.end(), std::minstd_rand(0) ); // use seed 0, so repeatable
    for( int i = 0; i < 10; i++ ) {
        cout << i << "=" << sequence[i] << endl;
    }
    // now sequence is shuffled, and has the numbers 0 to N, each exactly once
    // write out new, shuffled, training set
    int inputCubeSize = numPlanes * boardSize * boardSize;
    unsigned char *shuffledData = new unsigned char[N * inputCubeSize ];
    for( int n = 0; n < N; n++ ) {
        int newLocation = sequence[n];
        memcpy( &(shuffledData[newLocation*inputCubeSize]), &(training[n * inputCubeSize]), sizeof(unsigned char) * inputCubeSize );
    }
    //NorbLoader::writeImages( norbDir + "/training-shuffled-dat.mat", shuffledData, N, numPlanes, boardSize );
    delete[] shuffledData;

    return 0;
}

