// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


// this will take random data, and send it to stdout, which we can pipe into `predict`, to test predict's
// 'pipe' input

#include <iostream>
#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif // _WIN32
#include <random>

using namespace std;

int main( int argc, char *argv[] ) {
    const int numExamples = 9;
    const int planes = 1;
    const int size = 28;
    //int linearLength = numExamples * planes * size * size;
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
    for(int it=0; it < 10; it++) {
        for(int n=0; n < numExamples; n++) {
            for(int p = 0; p < planes; p++) {
                for(int h=0; h < size; h++) {
                    for(int w=0; w < size; w++) {
                        float random = ((float) rand()) / (float) RAND_MAX;
                        cout.write( reinterpret_cast< char * >( &random ), 1 * 4l );
                    }
                }
            }
        }
        cin.get();
    }

    return 0;
}

