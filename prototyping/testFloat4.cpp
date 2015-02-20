#include <iostream>
#include <string>
#include <cstring>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "Timer.h"
#include "test/TestArgsParser.h"
#include "test/SpeedTemplates.h"

using namespace std;

// for use in conjunction with http://www.cs.berkeley.edu/~volkov/volkov10-GTC.pdf
int main( int argc, char *argv[] ) {
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    int kernelVersion = 1;
    int numFloats = 16;
    int workgroupSize = 32;
    int numWorkgroups = 1;

    TestArgsParser args( argc, argv );
    args._arg( "n", &numFloats );
    args._arg( "workgroupsize", &workgroupSize );
    args._arg( "numworkgroups", &numWorkgroups );
    args._arg( "kernel", &kernelVersion );
    args._go();
    cout << "values:" << endl;
    args._printValues();

    string kernelSource1 = R"DELIM(
        kernel void memcpy( global float const*src, global float *dest) {
            if( get_global_id(0) < N ) {
                float a = src[get_global_id(0)];
                dest[get_global_id(0)] = a;
            }
        }
    )DELIM";

    string kernelSource2 = R"DELIM(
        kernel void memcpy( global float const*src, global float *dest) {
            global float4 *dest4 = (global float4*)&(dest[0]);
            global float4 *src4 = (global float4*)&(src[0]);
            int offset = get_global_id(0);
            if( offset < ( N >> 2 ) ) {
                float4 a = src4[offset];
                dest4[offset] = a;
            }
        }
    )DELIM";

    string options = "";
    options += " -cl-mad-enable";
    options += " -cl-single-precision-constant";
    options += " -D N=" + toString( numFloats );
    
    string kernelSource = "";
    if( kernelVersion == 1 ) {
        kernelSource = kernelSource1;
    } else if( kernelVersion == 2 ) {
        kernelSource = kernelSource2;
    } else {
        throw runtime_error("kernel version " + toString( kernelVersion ) + " not implemented");
    }
    
    CLKernel *kernel = cl->buildKernelFromString( kernelSource, "memcpy", options );

    float *src = new float[numFloats];
    for( int i = 0; i < numFloats; i++ ) {
        src[i] = i + 3.0f;
    }
    float *dest = new float[numFloats];
    memset( dest, 0, numFloats * sizeof(float) );
    kernel->in( numFloats, src );
    kernel->out( numFloats, dest );
//    int numWorkgroups = ( numFloats + workgroupSize - 1 ) / workgroupSize;
    Timer timer;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();
    float kernelTime = timer.lap();
    cout << "time: " << kernelTime << "ms" << endl;
    for( int i = 0; i < numFloats; i++ ) {
        if( i > 0 && i % 4 == 0 ) {
            cout << endl;
        }
        cout << " " << dest[i];
    }
    cout << endl;
//    cout << stuff[0] << " " << stuff[1] << endl;

    float kernelTimeSeconds = kernelTime / 1000.0f;

    delete[] src;
    delete[] dest;
    delete kernel;
    delete cl;

    return 0;
}

