// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

// create a data file in format appropriate for deepclexec, from mnist,
// which is small and easy to train

#include <iostream>

#include "loaders/GenericLoader.h"
#include "util/FileHelper.h"

using namespace std;

int main( int argc, char *argv[] ) {
    if( argc != 4 ) {
        cout << "Usage: " << argv[0] << " [mnist images file (input)] [floats file (output, overwritten)] [num examples]" << endl;
        cout << "note: for testing deepclexec, 1280 examples are probably 'good enough'" << endl;
        return 1;
    }
    string mnistImagesFile = argv[1];
    string outFile = argv[2];
    int numExamples = atoi(argv[3]);
    
    int N, planes, size;
    GenericLoader::getDimensions( mnistImagesFile.c_str(), &N, &planes, &size );
    float *imageData = new float[ N * planes * size * size ];
    int *labels = new int[N]; // we'll just throw this away, but it keeps the genericloader happy
                              // probably want an option to not load this actually...
    GenericLoader::load( mnistImagesFile.c_str(), imageData, labels, 0, numExamples ); 

    // now we've loaded the data, write it out in deepclexec-expecting format
    int linearLength = numExamples * planes * size * size;
    FileHelper::writeBinary( outFile, reinterpret_cast< char * >(imageData), linearLength * 4l );

    delete[] labels;
    delete[] imageData;

    return 0;
}

