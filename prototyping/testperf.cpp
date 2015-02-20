#include <iostream>
#include <string>
#include <cstring>

#include "OpenCLHelper.h"
#include "stringhelper.h"
#include "Timer.h"
#include "test/TestArgsParser.h"
#include "test/SpeedTemplates.h"

using namespace std;

int main( int argc, char *argv[] ) {
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    int its = 10000000;
    int workgroupSize = 512;
    bool optimizerOn = true;
    int count = 4;
//    int dOn = false;
    bool enableMad = true;

    TestArgsParser args( argc, argv );
    args._arg( "its", &its );
    args._arg( "workgroupsize", &workgroupSize );
    args._arg( "opt", &optimizerOn );
    args._arg( "count", &count );
//    args._arg( "don", &dOn );
    args._arg( "mad", &enableMad );
    args._go();

    cout << "its: " << its << " workgroupsize: " << workgroupSize << " optimizer: " << optimizerOn << endl;  

    string kernelSource = R"DELIM(
        kernel void test(global float *stuff) {
            float a = stuff[get_global_id(0)];
#ifdef dOn
            float d = stuff[get_global_id(0)+1024];
            float e = stuff[get_global_id(0)+1];
            float f = stuff[get_global_id(0)+2];
#endif
            float b = stuff[5000];
            float c = stuff[5001];
            #pragma unroll 100
            for( int i = 0; i < N_ITERATIONS; i++ ) {
                a = a * b + c;
#ifdef dOn
                d = d * b + c;
                e = e * b + c;
                f = f * b + c;
#endif
            }
            stuff[get_global_id(0)] = a;
#ifdef dOn
            stuff[get_global_id(0) + 1024] = d;
            stuff[get_global_id(0) + 1024] = e;
            stuff[get_global_id(0) + 1024] = f;
#endif
        }
    )DELIM";
    
    string kernelSource2 = R"DELIM(
        kernel void test(global float *stuff) {
            float a[COUNT];
            #pragma unroll COUNT
            for( int i = 0; i < COUNT; i++ ) {
                a[i] = stuff[get_global_id(0) + i ];
            }
            float b = stuff[5000];
            float c = stuff[5001];
            #pragma unroll 100
            for( int i = 0; i < N_ITERATIONS; i++ ) {
                #pragma unroll COUNT
                for( int j = 0; j < COUNT; j++ ) {
                    a[j] = a[j] * b + c;
                }
            }
            #pragma unroll COUNT
            for( int i = 0; i < COUNT; i++ ) {
                stuff[get_global_id(0) + 1] = a[i];
            }
        }
    )DELIM";
    
    string kernelSource3 = R"DELIM(
        kernel void test(global float *stuff) {
            float a[{{COUNT}}];
            {% for i in range( COUNT ) %}
                a[{{i}}] = stuff[get_global_id(0) + {{i}} ];{% endfor %}
            float b = stuff[5000];
            float c = stuff[5001];
            for( int it = 0; it < N_ITERATIONS / {{unroll}}; it++ ) {
                {% for i in range( unroll ) %}{% for j in range( COUNT ) %}
                        a[{{j}}] = a[{{j}}] * b + c;{% endfor %}{% endfor %}
            }
            {% for i in range( COUNT ) %}
                stuff[get_global_id(0) + 1] = a[{{i}}];{% endfor %}
        }
    )DELIM";
    
    string options = "";
    if( !optimizerOn ) {
        options += " -cl-opt-disable";
    }
    if( enableMad ) {
        options += " -cl-mad-enable";
    }
    options += " -D COUNT=" + toString( count );
//    if( dOn ) {
//        options += " -D dOn";
//    }
    options += " -cl-single-precision-constant";
    options += " -D N_ITERATIONS=" + toString( its );
    
    SpeedTemplates::Template mytemplate( kernelSource3 );
    mytemplate.setValue( "COUNT", count );
    mytemplate.setValue( "N_ITERATIONS", its );
    mytemplate.setValue( "unroll", 25 );
    string renderedSource3 = mytemplate.render();
    cout << "rendered source 3: [" << renderedSource3 << "]" << endl;
    CLKernel *kernel = cl->buildKernelFromString( renderedSource3, "test", options );

    float stuff[10000];
    for( int i = 0; i < 10000; i++ ) {
        stuff[i] = 2.0f;
    }
    stuff[5000] = 2.0f;
    stuff[5001] = -1.999999f;
    kernel->inout( 10000, stuff );
    Timer timer;
    kernel->run_1d( workgroupSize, workgroupSize );
    cl->finish();
    float kernelTime = timer.lap();
    cout << "time: " << kernelTime << "ms" << endl;
    cout << stuff[0] << " " << stuff[1] << endl;

    float kernelTimeSeconds = kernelTime / 1000.0f;

    float throughputGflops = (float)its * workgroupSize / kernelTimeSeconds / 1024 / 1024 / 1024;
//    if( dOn ) {
        throughputGflops *= count;
//    }
    cout << "throughput: " << throughputGflops << "Gflop/s" << endl;

    delete kernel;
    delete cl;

    return 0;
}

