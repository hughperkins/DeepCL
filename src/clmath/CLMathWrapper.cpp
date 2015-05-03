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
#include "clmath/GpuAdd.h"
#include "clmath/MultiplyInPlace.h"
#include "clmath/CLMathWrapper.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL CLMathWrapper::~CLMathWrapper() {
    delete copyBuffer;
    delete gpuAdd;
    delete multiplyInPlace;
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
    kernelPerElementMultInPlace->in( N )
            ->inout( wrapper )
            ->in( ((CLWrapper *)two.wrapper) );
    runKernel( kernelPerElementMultInPlace );
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator+=( const CLMathWrapper &two ) {
//    cout << "CLMathWrapper.operator+=()" << endl;
    if( two.N != N ) {
        throw runtime_error("CLMathWrapper::operator+, array size mismatch, cannot assign " + toString( two.N ) + 
            " vs " + toString( N ) );
    }
    gpuAdd->add( N, wrapper, ((CLMathWrapper &)two).wrapper );
    return *this;    
}
VIRTUAL CLMathWrapper &CLMathWrapper::operator=( const CLMathWrapper &rhs ) {
//    cout << "CLMathWrapper.operator=()" << endl;
    if( rhs.N != N ) {
        throw runtime_error("CLMathWrapper::operator= array size mismatch, cannot assign " + toString( rhs.N ) + 
            " vs " + toString( N ) );
    }
    copyBuffer->copy( N, ((CLMathWrapper &)rhs).wrapper, wrapper );
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::sqrt() {
    kernelSqrt->in( N )->inout( wrapper );
    runKernel( kernelSqrt );
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::inv() {
    kernelInv->in( N )->inout( wrapper );
    runKernel( kernelInv );
    return *this;
}
VIRTUAL CLMathWrapper &CLMathWrapper::squared() {
    kernelSquared->in( N )->inout( wrapper );
    runKernel( kernelSquared );
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
    this->copyBuffer = new CopyBuffer( cl );
    this->gpuAdd = new GpuAdd( cl );
    this->multiplyInPlace = new MultiplyInPlace( cl );

    buildSqrt();
    buildSquared();
    buildPerElementMultInPlace();
    buildAddScalar();
    buildInv();
}
void CLMathWrapper::buildInv() {
    std::string kernelName = "kernelInv";
    if( cl->kernelExists( kernelName ) ) {
        this->kernelInv = cl->getKernel( kernelName );
        return;
    }
    cout << "kernelInv: building kernel" << endl;

    string options = "";
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernelInv", "cl/inv.cl",
    //                          "array_inv", 'options' )
    // ]]]
    // generated using cog, from cl/inv.cl:
    const char * kernelInvSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "\n" 
    "// one over the value\n" 
    "\n" 
    "kernel void array_inv(\n" 
    "        const int N,\n" 
    "        global float *data ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    data[globalId] = 1.0f / data[globalId];\n" 
    "}\n" 
    "\n" 
    "";
    kernelInv = cl->buildKernelFromString( kernelInvSource, "array_inv", options, "cl/inv.cl" );
    // [[[end]]]
    cl->storeKernel( kernelName, kernelInv );
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
    cl->storeKernel( kernelName, kernelAddScalar );
}
void CLMathWrapper::buildPerElementMultInPlace() {
    std::string kernelName = "PerElementMultInPlace";
    if( cl->kernelExists( kernelName ) ) {
        this->kernelPerElementMultInPlace = cl->getKernel( kernelName );
        return;
    }
    cout << "PerElementMultInPlace: building kernel" << endl;

    string options = "";
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernelPerElementMultInPlace", "cl/per_element_mult.cl",
    //                          "per_element_mult_inplace", 'options' )
    // ]]]
    // generated using cog, from cl/per_element_mult.cl:
    const char * kernelPerElementMultInPlaceSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void per_element_mult_inplace( const int N, global float *target, global const float *source ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    target[globalId] *= source[globalId];\n" 
    "}\n" 
    "\n" 
    "";
    kernelPerElementMultInPlace = cl->buildKernelFromString( kernelPerElementMultInPlaceSource, "per_element_mult_inplace", options, "cl/per_element_mult.cl" );
    // [[[end]]]
    cl->storeKernel( kernelName, kernelPerElementMultInPlace );
}
void CLMathWrapper::buildSqrt() {
    std::string sqrtKernelName = "sqrt";
    if( cl->kernelExists( sqrtKernelName ) ) {
        this->kernelSqrt = cl->getKernel( sqrtKernelName );
        return;
    }
    cout << "sqrt: building kernel" << endl;

    string options = "";
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernelSqrt", "cl/sqrt.cl", "array_sqrt", 'options' )
    // ]]]
    // generated using cog, from cl/sqrt.cl:
    const char * kernelSqrtSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void array_sqrt(\n" 
    "        const int N,\n" 
    "        global float *data ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    data[globalId] = native_sqrt( data[globalId] );\n" 
    "}\n" 
    "\n" 
    "";
    kernelSqrt = cl->buildKernelFromString( kernelSqrtSource, "array_sqrt", options, "cl/sqrt.cl" );
    // [[[end]]]
    cl->storeKernel( sqrtKernelName, kernelSqrt );
}

void CLMathWrapper::buildSquared() {
    std::string kernelName = "squared";
    if( cl->kernelExists( kernelName ) ) {
        this->kernelSquared = cl->getKernel( kernelName );
        return;
    }
    cout << "squared: building kernel" << endl;

    string options = "";
    // [[[cog
    // import stringify
    // stringify.write_kernel2( "kernelSquared", "cl/squared.cl", "array_squared", 'options' )
    // ]]]
    // generated using cog, from cl/squared.cl:
    const char * kernelSquaredSource =  
    "// Copyright Hugh Perkins 2015 hughperkins at gmail\n" 
    "//\n" 
    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n" 
    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n" 
    "// obtain one at http://mozilla.org/MPL/2.0/.\n" 
    "\n" 
    "kernel void array_squared(\n" 
    "        const int N,\n" 
    "        global float *data ) {\n" 
    "    const int globalId = get_global_id(0);\n" 
    "    if( globalId >= N ) {\n" 
    "        return;\n" 
    "    }\n" 
    "    data[globalId] = data[globalId] * data[globalId];\n" 
    "}\n" 
    "\n" 
    "";
    kernelSquared = cl->buildKernelFromString( kernelSquaredSource, "array_squared", options, "cl/squared.cl" );
    // [[[end]]]
    cl->storeKernel( kernelName, kernelSquared );
}

