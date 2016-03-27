// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "test/PrintBuffer.h"
#include "test/CopyBuffer.h"

#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>

using namespace std;

void PrintBuffer::printFloats(EasyCL *cl, CLWrapper *buffer, int rows, int cols) {
    // first we will copy it to another buffer, so we can copy it out

    float *copiedBuffer = new float[ buffer->size() ];
    CopyBuffer::copy(cl, buffer, copiedBuffer);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            cout << " " << copiedBuffer[ i * cols + j ];
        }
        cout << endl;
    }

    delete[] copiedBuffer;
}

void PrintBuffer::printInts(EasyCL *cl, CLWrapper *buffer, int rows, int cols) {
    // first we will copy it to another buffer, so we can copy it out

    int *copiedBuffer = new int[ buffer->size() ];
    CopyBuffer::copy(cl, buffer, copiedBuffer);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            cout << " " << copiedBuffer[ i * cols + j ];
        }
        cout << endl;
    }

    delete[] copiedBuffer;
}

