// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"
#include "test/CopyBuffer.h"
#include "test/PrintBuffer.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

using namespace std;

TEST(testCopyBuffer, floats) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    const int N = 10;
    float *a = new float[N];
    for(int i = 0; i < N; i++) {
        a[i] = 3 + i;
    }
    CLWrapper *aWrapper = cl->wrap(N, a);
    aWrapper->copyToDevice();
    memset(a, 0, sizeof(float) * N);

    float *b = new float[N];
    CopyBuffer::copy(cl, aWrapper, b);
    
    for(int i = 0; i < N; i++) {
//        cout << b[i] << endl;
        EXPECT_EQ(3 + i, b[i]);
    }

    memset(b, 0, sizeof(float) * N);
    CopyBuffer::copy(cl, aWrapper, b);
    
    for(int i = 0; i < N; i++) {
//        cout << b[i] << endl;
        EXPECT_EQ(3 + i, b[i]);
    }

    PrintBuffer::printFloats(cl, aWrapper, 10, 1);
    
    delete[] b;
    delete aWrapper;
    delete[] a;

    delete cl;
}

TEST(testCopyBuffer, ints) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    const int N = 10;
    int *a = new int[N];
    for(int i = 0; i < N; i++) {
        a[i] = 3 + i;
    }
    CLWrapper *aWrapper = cl->wrap(N, a);
    aWrapper->copyToDevice();
    memset(a, 0, sizeof(int) * N);

    int *b = new int[N];
    CopyBuffer::copy(cl, aWrapper, b);
    
    for(int i = 0; i < N; i++) {
//        cout << b[i] << endl;
        EXPECT_EQ(3 + i, b[i]);
    }

    memset(b, 0, sizeof(int) * N);
    CopyBuffer::copy(cl, aWrapper, b);
    
    for(int i = 0; i < N; i++) {
//        cout << b[i] << endl;
        EXPECT_EQ(3 + i, b[i]);
    }

    PrintBuffer::printInts(cl, aWrapper, 10, 1);
    
    delete[] b;
    delete aWrapper;
    delete[] a;

    delete cl;
}

