// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


// this will take mnist data, and send it to stdout, which we can pipe into `predict`, to test predict's
// 'pipe' input

#include <iostream>
#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif // _WIN32

#include "loaders/GenericLoader.h"
#include "util/FileHelper.h"

using namespace std;

int main( int argc, char *argv[] ) {
    if( argc != 3 ) {
        cout << "Usage: " << argv[0] << " [mnist images file (input)] [num examples]" << endl;
        return 1;
    }
    string mnistImagesFile = argv[1];
    int numExamples = atoi(argv[2]);
    
    int N, planes, size;
    GenericLoader::getDimensions( mnistImagesFile.c_str(), &N, &planes, &size );
    float *imageData = new float[ N * planes * size * size ];
    int *labels = new int[N]; // we'll just throw this away, but it keeps the genericloader happy
                              // probably want an option to not load this actually...
    GenericLoader::load( mnistImagesFile.c_str(), imageData, labels, 0, numExamples ); 

    // now we've loaded the data, write it out to ... stdout?
    int linearLength = numExamples * planes * size * size;
    #ifdef _WIN32
    // refs:
    // http://www.thecodingforums.com/threads/binary-output-to-stdout-in-windows.317367/
    // http://www.cplusplus.com/forum/windows/77812/
    _setmode( _fileno( stdout ), _O_BINARY ); 
    #endif
    // I think we should at least write some kind of header, like how many planes etc...
    int dims[3];
    dims[0] = planes;
    dims[1] = size;
    dims[2] = size;
    cout.write( reinterpret_cast< char * >( dims ), 3 * 4l );
    cout.write( reinterpret_cast< char * >( imageData ), linearLength * 4l );
//    FileHelper::writeBinary( outFile, reinterpret_cast< char * >(imageData), linearLength * 4l );

    delete[] labels;
    delete[] imageData;

    return 0;
}


