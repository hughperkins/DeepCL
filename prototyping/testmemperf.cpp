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

//    int its = 10000000;
    int numFloats = 50000000;
    int workgroupSize = 512;
    bool optimizerOn = true;
    int count = 1;
//    bool enableMad = true;
    int kernelVersion = 1;

    TestArgsParser args( argc, argv );
//    args._arg( "its", &its );
    args._arg( "n", &numFloats );
    args._arg( "workgroupsize", &workgroupSize );
    args._arg( "opt", &optimizerOn );
    args._arg( "count", &count );
    args._arg( "kernel", &kernelVersion );
//    args._arg( "mad", &enableMad );
    args._go();
    cout << "values:" << endl;
    args._printValues();

    count = OpenCLHelper::getNextPower2( count );
    int thiscount = count;
    int shift = 0;
    while( thiscount > 1 ) {
        thiscount >>= 1;
        shift++;
    }
    cout << "count: " << count << " shift: " << shift << endl;

//    cout << "its: " << its << " workgroupsize: " << workgroupSize << " optimizer: " << optimizerOn << endl;  

    string kernelSource1 = R"DELIM(
        kernel void memcpy( global float const*src, global float *dest) {
            if( get_global_id(0) < N ) {
                float a = src[get_global_id(0)];
                dest[get_global_id(0)] = a;
            }
        }
    )DELIM";

    string kernelSource2 = R"DELIM(
        kernel void memcpy(global float const*src, global float *dest) {
            if( get_global_id(0) < N ) {
                dest[get_global_id(0)] = src[get_global_id(0)];
            }
        }
    )DELIM";

    string kernelSource3 = R"DELIM(
        kernel void memcpy(global float const*src, global float *dest) {
            dest[get_global_id(0)] = src[get_global_id(0)];
        }
    )DELIM";

    string kernelSource4 = R"DELIM(
        kernel void memcpy(global float const*src, global float *dest) {
            float a[COUNT];
            int offset = ( get_global_id(0) << SHIFT );
//            if( offset < N ) {
            if( ( offset + COUNT - 1 ) < N ) {
                #pragma unroll COUNT
                for( int i = 0; i < COUNT; i++ ) {
                    a[i] = src[ offset + i ];
                }
                #pragma unroll COUNT
                for( int i = 0; i < COUNT; i++ ) {
                    dest[ offset + i ] = a[i];
                }
            }
        }
    )DELIM";
    
    string kernelSource5 = R"DELIM(
        kernel void memcpy(global float const*src, global float *dest) {
            float a[{{COUNT}}];
            int offset = get_global_id(0) << {{SHIFT}};
            if( offset + {{COUNT}} - 1 < {{N}} ) {
                {% for i in range( COUNT ) %}
                    a[{{i}}] = src[ offset + {{i}} ];
                {% endfor %}
                {% for i in range( COUNT ) %}
                    dest[ offset + {{i}} ] = a[{{i}}];
                {% endfor %}
            }
        }
    )DELIM";
    
    string kernelSource6 = R"DELIM(
        #define f4(var) ( (global float4*)(var) )

        kernel void memcpy(global float const*src, global float *dest) {
            float4 a[COUNT];
            int offset = get_global_id(0) << SHIFT;
            if( get_global_id(0) < ( N >> SHIFT ) ) {
                #pragma unroll COUNT
                for( int i = 0; i < COUNT; i++ ) {
                    a[i] = f4(src)[ offset + i ];
                }
                #pragma unroll COUNT
                for( int i = 0; i < COUNT; i++ ) {
                    f4(dest)[ offset + i ] = a[i];
                }
            }
        }
    )DELIM";
    
    string kernelSource7 = R"DELIM(
        #define f4(var) ( (global float4*)(var) )

        kernel void memcpy(global float const*src, global float *dest) {
            float4 a[{{COUNT}}];
            int offset = get_global_id(0) << {{SHIFT}};
//            if( get_global_id(0) < ( {{N}} >> {{SHIFT}} ) ) {
            if( ( ( offset + {{COUNT}} - 1) << 2 ) < {{N}} ) {
                {% for i in range( COUNT ) %}
                    a[{{i}}] = f4(src)[ offset + {{i}} ];
                {% endfor %}
                {% for i in range( COUNT ) %}
                    f4(dest)[ offset + {{i}} ] = a[{{i}}];
                {% endfor %}
            }
        }
    )DELIM";

    string kernelSource8 = R"DELIM(
        kernel void memcpy(global float const*src, global float *dest) {
            local float a[{{COUNT}}];
            int offset = get_global_id(0) << {{SHIFT}};
//            if( get_global_id(0) < ( {{N}} >> {{SHIFT}} ) ) {
            if( ( ( offset + {{COUNT}} - 1) << 2 ) < {{N}} ) {
                {% for i in range( COUNT ) %}
                    a[{{i}}] = f4(src)[ offset + {{i}} ];
                {% endfor %}
                {% for i in range( COUNT ) %}
                    f4(dest)[ offset + {{i}} ] = a[{{i}}];
                {% endfor %}
            }
        }
    )DELIM";

    int numWorkgroups = 0;
    if( kernelVersion <= 5 ) {
        numWorkgroups = ( numFloats / count + workgroupSize - 1 ) / workgroupSize;
        numFloats = numWorkgroups * workgroupSize * count;
    } else {
        numWorkgroups = ( numFloats / count / 4 + workgroupSize - 1 ) / workgroupSize;
        numFloats = numWorkgroups * workgroupSize * 4 * count;
    }
    cout << "N=" << numFloats << " workgroupsize=" << workgroupSize << endl;

    string options = "";
    if( !optimizerOn ) {
        options += " -cl-opt-disable";
    }
//    if( enableMad ) {
        options += " -cl-mad-enable";
//    }
    options += " -D COUNT=" + toString( count );
    options += " -D SHIFT=" + toString( shift );
//    if( kernelVersion == 1 && count == 4 ) {
//        options += " -D dOn";
//    }
    options += " -cl-single-precision-constant";
//    options += " -D N_ITERATIONS=" + toString( its );
    options += " -D N=" + toString( numFloats );
//    options += " -D UNROLL=" + toString( unroll );
    
    string kernelSource = "";
    if( kernelVersion == 1 ) {
        kernelSource = kernelSource1;
    } else if( kernelVersion == 2 ) {
        kernelSource = kernelSource2;
    } else if( kernelVersion == 3 ) {
        kernelSource = kernelSource3;
    } else if( kernelVersion == 4 ) {
        kernelSource = kernelSource4;
    } else if( kernelVersion == 5 ) {
        SpeedTemplates::Template mytemplate( kernelSource5 );
        mytemplate.setValue( "COUNT", count );
        mytemplate.setValue( "N", numFloats );
        mytemplate.setValue( "SHIFT", shift );
//        mytemplate.setValue( "unroll", unroll );
        string renderedSource = mytemplate.render();
        cout << "rendered source: [" << renderedSource << "]" << endl;
        kernelSource = renderedSource;
    } else if( kernelVersion == 6 ) {
        kernelSource = kernelSource6;
    } else if( kernelVersion == 7 ) {
        SpeedTemplates::Template mytemplate( kernelSource7 );
        mytemplate.setValue( "N", numFloats );
        mytemplate.setValue( "COUNT", count );
        mytemplate.setValue( "SHIFT", shift );
//        mytemplate.setValue( "unroll", unroll );
        string renderedSource = mytemplate.render();
        cout << "rendered source: [" << renderedSource << "]" << endl;
        kernelSource = renderedSource;
    } else {
        throw runtime_error("kernel version " + toString( kernelVersion ) + " not implemented");
    }
    

    CLKernel *kernel = cl->buildKernelFromString( kernelSource, "memcpy", options );

    float *src = new float[numFloats];
    for( int i = 0; i < numFloats; i++ ) {
        src[i] = 2.0f;
    }
    float *dest = new float[numFloats + count];
    memset( dest, 0, ( numFloats + count ) * sizeof(float) );

    CLWrapper *destWrapper = cl->wrap( numFloats + count, dest );
    CLWrapper *srcWrapper = cl->wrap( numFloats + count, src );
    destWrapper->createOnDevice();
    srcWrapper->copyToDevice();
    kernel->in( srcWrapper );
    kernel->out( destWrapper );
    cl->finish(); // just in case...
    Timer timer;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();
    float kernelTime = timer.lap();
    cout << "time: " << kernelTime << "ms" << endl;
//    cout << stuff[0] << " " << stuff[1] << endl;

    float kernelTimeSeconds = kernelTime / 1000.0f;

    float throughputGbytes = numFloats * 4.0f  / 1024 / 1024 / 1024;
//    if( kernelVersion >= 4 ) {
//        throughputGbytes *= count;
//    }
    float throughputGbytespersec = throughputGbytes / kernelTimeSeconds;
//    float throughputGflops = (float)its * workgroupSize / kernelTimeSeconds / 1024 / 1024 / 1024;

    cout << "throughput: " << throughputGbytespersec << "GB/s" << endl;

    // check the memory copied...
    destWrapper->copyToHost();
    int errCount = 0;
    for( int i = 0; i < numFloats; i++ ) {
        if( dest[i] != 2.0f ) {
            if( errCount > 5 ) {
                cout << "..." << endl;
                break;
            }
            cout << "DIFF: dest[" << i << "]=" << dest[i] << endl;
            errCount++;
        }
    }
    for( int i = 0; i < count; i++ ) {
        if( dest[i + numFloats] != 0.0f ) {
            if( errCount > 5 ) {
                cout << "..." << endl;
                break;
            }
            cout << "DIFF: dest[" << ( i + numFloats ) << "]=" << dest[i + numFloats] << endl;
            errCount++;
        }
    }

    delete destWrapper;
    delete srcWrapper;
    delete[] src;
    delete[] dest;
    delete kernel;
    delete cl;

    return 0;
}

