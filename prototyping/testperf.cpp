#include <iostream>
#include <string>
#include <cstring>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "Timer.h"

using namespace std;

int main( int argc, char *argv[] ) {
    int its = atoi(argv[1] );
    int workgroupSize = atoi( argv[2] );

    cout << "its: " << its << " workgroupsize: " << workgroupSize << endl;  

    string kernelSource = R"DELIM(
        kernel void test(global float *stuff) {
            float a = stuff[get_global_id(0)];
            float b = stuff[1024];
            float c = stuff[1025];
            #pragma unroll 100
            for( int i = 0; i < N_ITERATIONS; i++ ) {
                a = a * b + c;
            }
            stuff[get_global_id(0)] = a;
        }
    )DELIM";
    
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();
    CLKernel *kernel = cl->buildKernelFromString( kernelSource, "test", "-cl-opt-disable -D N_ITERATIONS=" + toString( its ) );

    Timer timer;
    float stuff[2048];
    for( int i = 0; i < 2048; i++ ) {
        stuff[i] = 2.0f;
    }
    stuff[1024] = 2.0f;
    stuff[1025] = -1.999999f;
    kernel->inout( 2048, stuff );
    kernel->run_1d( workgroupSize, workgroupSize );
    cl->finish();
    float kernelTime = timer.lap();
    cout << "time: " << kernelTime << "ms" << endl;
    cout << stuff[0] << " " << stuff[1] << endl;

    float throughputGflops = (float)its * workgroupSize / kernelTime * 1000.0f / 1024 / 1024 / 1024;
    cout << "throughput: " << throughputGflops << "Gflop/s" << endl;

    delete kernel;
    delete cl;

    return 0;
}

