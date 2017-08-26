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

namespace testCopyBlock{

CLKernel *makeTestPosKernel( EasyCL *cl );
CLKernel *makeBasicKernel( EasyCL *cl );

TEST( testCopyBlock, testPos ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float in[12];
    float res[12];
    in[0] = ( 3 << 10 ) | 4;
    in[1] = 8;
    in[2] = 14;

    for( int i = 0; i < 3; i++ ) {
        cout << "in[" << i << "]=" << in[i] << endl;
    }
    
    CLKernel *kernel = makeTestPosKernel( cl );
    kernel->in( 12, in )->out( 12, res );
    kernel->run_1d(1,1); 
    cl->finish();

    for( int i = 0; i < 5; i++ ) {
        cout << "res[" << i << "]=" << res[i] << endl;
    }
}

//int posToRow( int pos ) {
//    return ( pos >> 10 ) & ( 2^11-1);
//}
//int posToCol( int pos ) {
//    return pos & (2^11-1);
//}
//int rowColToPos( int row, int col ) {
//    return ( row << 10 ) | col;
//}
//int linearIdToPos( int linearId, int base ) {
//    return rowColToPos( ( linearId / base ), ( linearId % base )  );
//}
//int posToOffset( int pos, int rowLength ) {
//    return posToRow(pos) * rowLength + posToCol(pos);
//}

TEST( testCopyBlock, basic ) {
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();

    float a[] = { 1,2,3,4,
                  5,6,7,8,
                  9,10,11,12 };
    float b[10];
    memset(b, 0, sizeof(float)*10);

    CLKernel *kernel = makeBasicKernel( cl );
    kernel->in( 12, a )->out( 6, b )->in( ( 3<<10)|4)->in( (0<<10)|1)->in((2<<10)|3);
    kernel->localFloats( 2 * 3 );
    kernel->run_1d(12,4);
//    kernel->run_1d(12,12);
    cl->finish();
    float expected[] = { 2,3,4,
                         6,7,8 }; 
    for( int i = 0; i < 2; i++ ) {
        for( int j = 0; j < 3; j++ ) {
            cout << b[i*3+j] << " ";
            EXPECT_EQ( expected[i*3+j], b[i*3+j] );
        }
        cout << endl;
    }
    cout << endl;
    for( int i = 6; i < 10; i++ ) {
        cout << b[i] << " ";
        EXPECT_EQ( 0, b[i] );
    }
    cout << endl;
        cout << endl;

    kernel->in( 12, a )->out( 6, b )->in( ( 3<<10)|4)->in( (1<<10)|0)->in((2<<10)|3);
    kernel->localFloats( 2 * 3 );
//    kernel->run_1d(12,4);
    kernel->run_1d(12,4);
    cl->finish();
    float expected2[] = { 5,6,7,
                         9,10,11 }; 
    for( int i = 0; i < 2; i++ ) {
        for( int j = 0; j < 3; j++ ) {
            cout << b[i*3+j] << " ";
            EXPECT_EQ( expected2[i*3+j], b[i*3+j] );
        }
        cout << endl;
    }
    cout << endl;
    for( int i = 6; i < 10; i++ ) {
        cout << b[i] << " ";
        EXPECT_EQ( 0, b[i] );
    }
    cout << endl;
        cout << endl;

    delete kernel;
    delete cl;
}

CLKernel *makeTestPosKernel( EasyCL *cl ) {
    CLKernel *kernel = 0;
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "test/testCopyBlock.cl", "testPos", '""' )
    // ]]]
    // generated using cog, from test/testCopyBlock.cl:
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
    "kernel void testPos( global const float *in, global float *out ) {\n"
    "    if( get_global_id(0) == 0 ) {\n"
    "        out[0] = posToRow( in[0] );\n"
    "        out[1] = posToCol( in[0] );\n"
    "        int pos = rowColToPos( in[1], in[2] );\n"
    "        out[2] = pos;\n"
    "        out[3] = posToRow(pos);\n"
    "        out[4] = posToCol(pos);\n"
    "    }\n"
    "}\n"
    "\n"
    "kernel void run( global const float *source, global float *destination, int sourceSize, int blockPos, int blockSize,\n"
    "    local float *localBuffer ) {\n"
    "    copyBlock( localBuffer, source, sourceSize, blockPos, blockSize );\n"
    "    //copyLocal( localBuffer, source, posToRow( blockSize ) * posToCol( blockSize ) );\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    copyGlobal( destination, localBuffer, posToRow( blockSize ) * posToCol( blockSize ) );\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "testPos", "", "test/testCopyBlock.cl");
    // [[[end]]]

    return kernel;
}

CLKernel *makeBasicKernel( EasyCL *cl ) {
    CLKernel *kernel = 0;
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernel", "test/testCopyBlock.cl", "run", '""' )
    // ]]]
    // generated using cog, from test/testCopyBlock.cl:
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
    "kernel void testPos( global const float *in, global float *out ) {\n"
    "    if( get_global_id(0) == 0 ) {\n"
    "        out[0] = posToRow( in[0] );\n"
    "        out[1] = posToCol( in[0] );\n"
    "        int pos = rowColToPos( in[1], in[2] );\n"
    "        out[2] = pos;\n"
    "        out[3] = posToRow(pos);\n"
    "        out[4] = posToCol(pos);\n"
    "    }\n"
    "}\n"
    "\n"
    "kernel void run( global const float *source, global float *destination, int sourceSize, int blockPos, int blockSize,\n"
    "    local float *localBuffer ) {\n"
    "    copyBlock( localBuffer, source, sourceSize, blockPos, blockSize );\n"
    "    //copyLocal( localBuffer, source, posToRow( blockSize ) * posToCol( blockSize ) );\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    copyGlobal( destination, localBuffer, posToRow( blockSize ) * posToCol( blockSize ) );\n"
    "}\n"
    "\n"
    "";
    kernel = cl->buildKernelFromString(kernelSource, "run", "", "test/testCopyBlock.cl");
    // [[[end]]]

    return kernel;
}

}

