// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "clmath/GpuOp.h"

#include "gtest/gtest.h"
#include "test/gtest_supp.h"
#include "test/WeightRandomizer.h"

using namespace std;

TEST( testGpuOp, addinplace ) {
    EasyCL *cl = new EasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    float bdat[] = { 4,2.1f, 5,3,9.2f };
    CLWrapper *a = cl->wrap( 5,adat );
    CLWrapper *b = cl->wrap( 5,bdat );
    a->copyToDevice();
    b->copyToDevice();

    GpuOp gpuOp( cl );
    gpuOp.apply( 5, a, b, new Op2Add() );
    a->copyToHost();

    for( int i = 0; i < 5; i++ ) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR( 5.0f, adat[0] );
    EXPECT_FLOAT_NEAR( 5.1f, adat[1] );
    EXPECT_FLOAT_NEAR( 2.5f + 9.2f, adat[4] );

//    delete a;
//    delete b;
    delete a;
    delete b;
    delete cl;
}


