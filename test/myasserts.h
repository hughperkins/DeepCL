// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

//#include "MyException.h"

#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>

template<typename T>
std::string toString(T val ) { // not terribly efficient, but works...
   std::ostringstream myostringstream;
   myostringstream << val;
   return myostringstream.str();
}
void assertEquals( float one, float two, float tolerance ) {
   float absdiff = one - two > 0 ? one - two : two - one;
   if( absdiff > tolerance ) {
      throw std::runtime_error( "assertEquals fail " + toString(one) + " != " + toString(two) );
   }
}

template<typename T>
void assertEquals( T one, T two ) {
    if( one != two ) {
        throw std::runtime_error( "assertEquals fail " + toString(one) + " != " + toString(two) );
    }
}

void assertEquals( float one, int two ) {
   assertEquals( one, (float)two );
}
void assertEquals( int one, float two ) {
   assertEquals( (float)one, two );
}

void assertLessThan( float expected, float actual ) {
    if( expected <= actual ) {
        throw std::runtime_error( "assertLessThan fail expected max " + toString(expected) + " but was " + toString(actual) );
    }
}

void assertEquals( int one, int two ) {
   assertEquals<int>( one, two );
}

