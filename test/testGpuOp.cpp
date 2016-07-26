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
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    float bdat[] = { 4,2.1f, 5,3,9.2f };
    CLWrapper *a = cl->wrap( 5,adat );
    CLWrapper *b = cl->wrap( 5,bdat );
    a->copyToDevice();
    b->copyToDevice();

    GpuOp gpuOp( cl );
    gpuOp.apply2_inplace( 5, a, b, new Op2Add() );
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

TEST( testGpuOp, addoutofplace ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    float bdat[] = { 4,2.1f, 5,3,9.2f };
    float cdat[5];
    CLWrapper *a = cl->wrap( 5,adat );
    CLWrapper *b = cl->wrap( 5,bdat );
    CLWrapper *c = cl->wrap( 5,cdat );
    a->copyToDevice();
    b->copyToDevice();
    c->copyToDevice();

    GpuOp gpuOp( cl );
    gpuOp.apply2_outofplace( 5, c, a, b, new Op2Add() );
    a->copyToHost();
    c->copyToHost();

    for( int i = 0; i < 5; i++ ) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    for( int i = 0; i < 5; i++ ) {
        cout << "c[" << i << "]=" << cdat[i] << endl;
    }
    EXPECT_FLOAT_NEAR( 1.0f, adat[0] );
    EXPECT_FLOAT_NEAR( 3.0f, adat[1] );
    EXPECT_FLOAT_NEAR( 2.5f, adat[4] );

    EXPECT_FLOAT_NEAR( 5.0f, cdat[0] );
    EXPECT_FLOAT_NEAR( 5.1f, cdat[1] );
    EXPECT_FLOAT_NEAR( 2.5f + 9.2f, cdat[4] );

//    delete a;
//    delete b;
    delete a;
    delete b;
    delete cl;
}

TEST( testGpuOp, inverse ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    CLWrapper *a = cl->wrap( 5,adat );
    a->copyToDevice();

    GpuOp gpuOp( cl );
    gpuOp.apply1_inplace( 5, a, new Op1Inv() );
    a->copyToHost();

    for( int i = 0; i < 5; i++ ) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR( 1, adat[0] );
    EXPECT_FLOAT_NEAR( 0.333333f, adat[1] );
    EXPECT_FLOAT_NEAR( 1.0f / 9.0f, adat[2] );
    EXPECT_FLOAT_NEAR( 1.0f / 2.5f, adat[4] );

    delete a;
    delete cl;
}

TEST( testGpuOp, addscalarinplace ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    float adat[] = { 1,3,9,12.5f,2.5f };
    CLWrapper *a = cl->wrap( 5,adat );
    a->copyToDevice();

    GpuOp gpuOp( cl );
    gpuOp.apply2_inplace( 5, a, 4.2f, new Op2Add() );
    a->copyToHost();

    for( int i = 0; i < 5; i++ ) {
        cout << "a[" << i << "]=" << adat[i] << endl;
    }
    EXPECT_FLOAT_NEAR( 5.2f, adat[0] );
    EXPECT_FLOAT_NEAR( 7.2f, adat[1] );
    EXPECT_FLOAT_NEAR( 2.5f + 4.2f, adat[4] );

    delete a;
    delete cl;
}


