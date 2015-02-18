#include <iostream>
#include <string>

#include "stringhelper.h"
#include "OpenCLHelper.h"

using namespace std;

int main( int argc, char *argv[] ) {
    OpenCLHelper *cl = OpenCLHelper::createForFirstGpuOtherwiseCpu();

    const string kernelSource = R"DELIM(

typedef struct _MyStruct {
    float a;
    float b;
    int c;
    int d;
} MyStruct;

constant MyStruct myStruct = mystruct_init;
//constant MyStruct myStruct = { .a = 1.23f };
//constant MyStruct myStruct = { .a = 1.23f, .b = 5.67f, .c = 8, .d = 4 };

void doSomething( constant MyStruct *s, global float *floats ) {
        floats[0] = s->a;
        floats[1] = s->b;
}

void copyLocal( local float *target, global float const *source, int N ) {
    int numLoops = ( N + workgroupSize - 1 ) / workgroupSize;
    for( int loop = 0; loop < numLoops; loop++ ) {
        int offset = loop * workgroupSize;
        if( offset < N ) {
            target[offset + get_local_id(0)] = source[offset + get_local_id(0)];
        }
    }
}

void copyGlobal( global float *target, local float const*source, int N ) {
    int numLoops = ( N + workgroupSize - 1 ) / workgroupSize;
    for( int loop = 0; loop < numLoops; loop++ ) {
        int offset = loop * workgroupSize;
        if( offset < N ) {
            target[offset + get_local_id(0)] = source[offset + get_local_id(0)];
        }
    }
}

kernel void myKernel( global float *result, global int *ints, local float*localFloats, global float *a ) {
    if( get_global_id(0) == 0 ) {
        ints[0] = myStruct.c;
        ints[1] = myStruct.d;
        doSomething( &myStruct, result );
    }
    copyLocal( localFloats, a, 10 );
    if( get_local_id(0) < 10 ) {
        localFloats[get_local_id(0)] = 3 + get_local_id(0);
    }
    copyGlobal( a, localFloats, 10 );
}
 
    )DELIM";

    int workgroupSize = 32;
    CLKernel *kernel = cl->buildKernelFromString( kernelSource, "myKernel", "-D mystruct_init={.a=4.56f,.b=3.12f,.c=12,.d=34} -D workgroupSize=" + toString(workgroupSize) );

    float floats[10];
    int ints[10];
    float c[10];
    kernel->out( 10, floats )->out(10, ints )->localFloats(10)->out(10, c)->run_1d( workgroupSize, workgroupSize );
    cl->finish();
    cout << floats[0] << endl;
    cout << floats[1] << endl;
    cout << ints[0] << endl;
    cout << ints[1] << endl;
    cout << c[0] << endl;
    cout << c[1] << endl;
    cout << c[2] << endl;

    delete kernel;

    delete cl;
    return 0;
}

