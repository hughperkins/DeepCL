// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "EasyCL.h"
#include "CLFloatWrapper.h"
#include "util/stringhelper.h"
#include "clmath/CopyBuffer.h"
#include "clmath/GpuOp.h"
#include "clmath/MultiplyInPlace.h"
#include "clmath/CLMathWrapper.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL CLMathWrapper::~CLMathWrapper() {
    delete multiplyInPlace;
    delete gpuOp;
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator*=( const float scalar ) {
//    cout << "CLMathWrapper.operator*=(scalar)" << endl;
    multiplyInPlace->multiply( N, scalar, wrapper );
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator+=( const float scalar ) {
//    cout << "CLMathWrapper.operator*=(scalar)" << endl;
    kernelAddScalar->in( N )->in( scalar )->inout( wrapper );
    runKernel( kernelAddScalar );
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator*=( const CLMathWrapper &two ) {
//    cout << "CLMathWrapper.operator*=(scalar)" << endl;
    if( two.N != N ) {
        throw runtime_error("CLMathWrapper::operator+, array size mismatch, cannot assign " + toString( two.N ) + 
            " vs " + toString( N ) );
    }
    Op2Mul op;
    gpuOp->apply2_inplace( N, wrapper, ((CLMathWrapper &)two).wrapper, &op );
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator+=( const CLMathWrapper &two ) {
//    cout << "CLMathWrapper.operator+=()" << endl;
    if( two.N != N ) {
        throw runtime_error("CLMathWrapper::operator+, array size mismatch, cannot assign " + toString( two.N ) + 
            " vs " + toString( N ) );
    }
    Op2Add op;
    gpuOp->apply2_inplace( N, wrapper, ((CLMathWrapper &)two).wrapper, &op );
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator=( const CLMathWrapper &rhs ) {
//    cout << "CLMathWrapper.operator=()" << endl;
    if( rhs.N != N ) {
        throw runtime_error("CLMathWrapper::operator= array size mismatch, cannot assign " + toString( rhs.N ) + 
            " vs " + toString( N ) );
    }
    Op1Equal op;
    gpuOp->apply1_outofplace( N, wrapper, ((CLMathWrapper &)rhs).wrapper, &op );
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::sqrt() {
    Op1Sqrt op;
    gpuOp->apply1_inplace( N, wrapper, &op );
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::inv() {
    Op1Inv op;
    gpuOp->apply1_inplace( N, wrapper, &op );
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::squared() {
    Op1Squared op;
    gpuOp->apply1_inplace( N, wrapper, &op );
    return *this;
}
VIRTUAL void CLMathWrapper::runKernel( CLKernel *kernel ) {   
    int globalSize = N;
    int workgroupSize = 64;
    int numWorkgroups = ( globalSize + workgroupSize - 1 ) / workgroupSize;
    kernel->run_1d( numWorkgroups * workgroupSize, workgroupSize );
    cl->finish();
}
CLMathWrapper::CLMathWrapper( CLWrapper *wrapper ) {
    CLFloatWrapper *floatWrapper = dynamic_cast< CLFloatWrapper * >( wrapper );
    if( floatWrapper == 0 ) {
        throw runtime_error( "CLMathWrapper only works on CLFloatWrapper objects");
    }
    this->cl = floatWrapper->getCl();
    this->wrapper = floatWrapper;
    this->N = floatWrapper->size();
    this->multiplyInPlace = new MultiplyInPlace( cl );
    this->gpuOp = new GpuOp( cl );

    buildAddScalar();
}
void CLMathWrapper::buildAddScalar() {
    std::string kernelName = "kernelAddScalar";
    if( cl->kernelExists( kernelName ) ) {
        this->kernelAddScalar = cl->getKernel( kernelName );
        return;
    }
    cout << "kernelAddScalar: building kernel" << endl;

    string options = "";
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernelAddScalar", "cl/addscalar.cl",
    //                          "add_scalar", 'options' )
    // ]]]
    // generated using cog, from cl/addscalar.cl:
    const char * kernelAddScalarSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void add_scalar(\n" 
    "        const int N,\n" 
    "        const float scalar,\n" 
    "        global float *data ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    data[globalId] += scalar;\n" 
    "}\n" 
    "\n" 
    "";
    kernelAddScalar = cl->buildKernelFromString( kernelAddScalarSource, "add_scalar", options, "cl/addscalar.cl" );
    // [[[end]]]
    cl->storeKernel( kernelName, kernelAddScalar, true );
}

