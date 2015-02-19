#include <iostream>
#include <string>
#include <cstring>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "Timer.h"
#include "test/TestArgsParser.h"

using namespace std;

int main( int argc, char *argv[] ) {
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    int its = 10000000;
    int workgroupSize = 512;
    bool optimizerOn = true;

    TestArgsParser args( argc, argv );
    args._arg( "its", &its );
    args._arg( "workgroupsize", &workgroupSize );
    args._arg( "opt", &optimizerOn );
    args._go();

    cout << "its: " << its << " workgroupsize: " << workgroupSize << " optimizer: " << optimizerOn << endl;  

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
    
    string optimizerDefine = optimizerOn ? "" : "-cl-opt-disable ";
    CLKernel *kernel = cl->buildKernelFromString( kernelSource, "test", optimizerDefine + " -D N_ITERATIONS=" + toString( its ) );

    float stuff[2048];
    for( int i = 0; i < 2048; i++ ) {
        stuff[i] = 2.0f;
    }
    stuff[1024] = 2.0f;
    stuff[1025] = -1.999999f;
    kernel->inout( 2048, stuff );
    Timer timer;
    kernel->run_1d( workgroupSize, workgroupSize );
    cl->finish();
    float kernelTime = timer.lap();
    cout << "time: " << kernelTime << "ms" << endl;
    cout << stuff[0] << " " << stuff[1] << endl;

    float kernelTimeSeconds = kernelTime / 1000.0f;
    float throughputGflops = (float)its * workgroupSize / kernelTimeSeconds / 1024 / 1024 / 1024;
    cout << "throughput: " << throughputGflops << "Gflop/s" << endl;

    delete kernel;
    delete cl;

    return 0;
}

