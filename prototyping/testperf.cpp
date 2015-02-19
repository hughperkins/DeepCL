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
            float b = stuff[1];
            float c = stuff[2];
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
    stuff[1] = 2.0f;
    stuff[2] = -2.0f;
    kernel->inout( 2048, stuff );
    kernel->run_1d( workgroupSize, workgroupSize );
    cl->finish();
    timer.timeCheck("After kernel");
    cout << stuff[0] << " " << stuff[1] << endl;

    delete kernel;
    delete cl;

    return 0;
}

