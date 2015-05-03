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
}
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

