// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include "EasyCL.h"
#include "templated/TemplatedKernel.h"

using namespace std;

TEST( testTemplatedKernel, basic ) {
    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();

    string kernelSource = "kernel void doStuff( int N, global {{type}} *out, global const {{type}} *in ) {\n"
        "   int globalId = get_global_id(0);\n"
        "   if( globalId < N ) {\n"
        "       {{type}} value = in[globalId];\n"
        "       out[globalId] = value;\n"
        "   }\n"
        "}\n";
    TemplatedKernel kernela(cl, "testfile", kernelSource, "doStuff");
    kernela.setValue("type", "int");
    CLKernel *kernel = kernela.getKernel();
    kernel = kernela.getKernel();
    int a[2];
    int b[2];
    b[0] = 3;
    b[1] = 2;
    kernel->in(2)->out(2, a)->in(2, b)->run_1d(16, 16);
    cl->finish();
    EXPECT_EQ(3, a[0]);
    EXPECT_EQ(2, a[1]);

    TemplatedKernel kernelb(cl, "testfile", kernelSource, "doStuff");
    kernelb.setValue("type", "float");
    kernel = kernelb.getKernel();
    float ac[2];
    float bc[2];
    bc[0] = 3.2f;
    bc[1] = 2.5f;
    kernel->in(2)->out(2, ac)->in(2, bc)->run_1d(16, 16);
    cl->finish();
    EXPECT_EQ(3.2f, ac[0]);
    EXPECT_EQ(2.5f, ac[1]);

    delete cl;
}

TEST( testTemplatedKernel, basic2 ) {
    EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();

    string kernelSource = "kernel void doStuff( global int *value) {\n"
        "   int globalId = get_global_id(0);\n"
        "   if( globalId == 0 ) {\n"
        "       value[0] = {{myvalue}};\n"
        "   }\n"
        "}\n";
    TemplatedKernel kernela(cl, "testfile", kernelSource, "doStuff");
    for( int i = 0; i < 10; i++ ) {
        kernela.setValue("myvalue", i );
        CLKernel *kernel = kernela.getKernel();
        kernel = kernela.getKernel();
        int a[1];
        kernel->out(1, a)->run_1d(16, 16);
        cl->finish();
        cout << "i=" << i << " a[0]=" << a[0] << endl;
        EXPECT_EQ(i, a[0]);
    }

    delete cl;
}

