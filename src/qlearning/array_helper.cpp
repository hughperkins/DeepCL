// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "qlearning/array_helper.h"
#include "util/stringhelper.h"

using namespace std;

void arrayCopy(float *dest, float const*src, int N) {
    for(int i = 0; i < N; i++) {
        dest[i] = src[i];
    }
}

void arrayZero(float *array, int N) {
    for(int i = 0; i < N; i++) {
        array[i] = 0;
    }
}

string toString(float const*array, int N) {
    string result = "";
    for(int i = 0; i < N; i++) {
        result += toString(array[i]);
    }
    return result;
}


