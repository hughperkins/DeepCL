// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "gtest/gtest.h"
#include "test/gtest_supp.h"

#include <iostream>
#include <string>
#include <algorithm>

#include "EasyCL.h"

using namespace std;

namespace testCopyLocal{

CLKernel *makeKernel( EasyCL *cl );

TEST( testCopyLocal, basic ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float a[] = { 1,2,3,4,
                  5,6,7,8,
                  9,10,11,12 };
    float b[16];
    memset(b, 0, sizeof(float)*16);
    for( int i = 12; i < 16; i++ ) {
        cout << b[i] << " ";
        EXPECT_FLOAT_NEAR( 0, b[i] );
    }
    cout << endl;

    CLKernel *kernel = makeKernel( cl );
    kernel->in( 12, a )->inout( 16, b )->in( 12 );
    kernel->localFloats( 12 );
    kernel->run_1d(12,12);
    cl->finish();
    for( int i = 0; i < 3; i++ ) {
        for( int j = 0; j < 4; j++ ) {
            cout << b[i*4+j] << " ";
            EXPECT_EQ( i * 4 + j + 1, b[i*4+j] );
        }
        cout << endl;
    }
    cout << endl;
    for( int i = 12; i < 16; i++ ) {
        cout << b[i] << " ";
        EXPECT_FLOAT_NEAR( 0, b[i] );
    }
    cout << endl;

    delete kernel;
    delete cl;
}

CLKernel *makeKernel( EasyCL *cl ) {
    CLKernel *kernel = 0;
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "test/testCopyLocal.cl", "run", '""' )
    // ]]]
    // generated using cog, from test/testCopyLocal.cl:
    const char * kernelSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "// including cl/copyBlock.cl:\n"
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "static int posToRow(int pos) {\n"
    "    return (pos >> 10) & ((1<<10)-1);\n"
    "//    return 53\n"
    "}\n"
    "static int posToCol(int pos) {\n"
    "    return pos & ((1<<10)-1);\n"
    "  //  return 67;\n"
    "    //return ((1<<11)-1);\n"
    "}\n"
    "static int rowColToPos(int row, int col) {\n"
    "    return (row << 10) | col;\n"
    "}\n"
    "static int linearIdToPos(int linearId, int base) {\n"
    "    return rowColToPos(( linearId / base), (linearId % base)  );\n"
    "}\n"
    "static int posToOffset(int pos, int rowLength) {\n"
    "    return posToRow(pos) * rowLength + posToCol(pos);\n"
    "}\n"
    "\n"
    "// assumes that the block will fit exactly into the target\n"
    "static void copyBlock(local float *target, global float const *source,\n"
    "    const int sourceSize, const int blockStart, const int blockSize) {\n"
    "    const int totalLinearSize = posToRow(blockSize) * posToCol(blockSize);\n"
    "    const int numLoops = (totalLinearSize + get_local_size(0) - 1) / get_local_size(0);\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        const int offset = get_local_id(0) + loop * get_local_size(0);\n"
    "        if (offset < totalLinearSize) {\n"
    "            const int offsetAsPos = linearIdToPos(offset, posToCol(blockSize) );\n"
    "            target[ offset ] = source[ posToOffset(blockStart + offsetAsPos, posToCol(sourceSize) ) ];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "\n"
    "\n"
    "// including cl/ids.cl:\n"
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "#define globalId (get_global_id(0))\n"
    "#define localId (get_local_id(0)  )\n"
    "#define workgroupId (get_group_id(0))\n"
    "#define workgroupSize (get_local_size(0))\n"
    "\n"
    "\n"
    "\n"
    "// including cl/copyLocal.cl:\n"
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
    "//\n"
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    "\n"
    "static void copyLocal(local float *target, global float const *source, int N) {\n"
    "    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n"
    "        if (offset < N) {\n"
    "            target[offset] = source[offset];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "static void copyGlobal(global float *target, local float const *source, int N) {\n"
    "    int numLoops = (N + get_local_size(0) - 1) / get_local_size(0);\n"
    "    for (int loop = 0; loop < numLoops; loop++) {\n"
    "        int offset = loop * get_local_size(0) + get_local_id(0);\n"
    "        if (offset < N) {\n"
    "            target[offset] = source[offset];\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "\n"
    "\n"
    "kernel void run( global const float *source, global float *destination, int N,\n"
    "    local float *localBuffer ) {\n"
    "    copyLocal( localBuffer, source, N );\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    copyGlobal( destination, localBuffer, N );\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "run", "", "test/testCopyLocal.cl");
    // [[[end]]]

    return kernel;
}

}

