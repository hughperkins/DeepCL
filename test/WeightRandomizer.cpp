// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <random>
#include <stdexcept>

#include "test/WeightRandomizer.h"
#include "ClConvolve_typedefs.h"

using namespace std;
using namespace ClConvolve;

//#if (_MSC_VER == 1500 || _MSC_VER == 1600  )
//#define TR1RANDOM
//typedef std::tr1::mt19937 MT19937;
//#else
//typedef std::mt19937 MT19937;
//#endif

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC void WeightRandomizer::randomize( int seed, float *values, int numValues, float minvalue, float maxvalue ) {
	
    MT19937 random;
    if(seed == 0) {throw std::runtime_error("seed should not be zero"); } // for windows
    random.seed((unsigned long)seed); // so always gives same output
    randomize( random, values, numValues, minvalue, maxvalue );
}
STATIC void WeightRandomizer::randomize( MT19937 &random, float *values, int numValues, float minvalue, float maxvalue ) {
    for( int i = 0; i < numValues; i++ ) {
        values[i] = random() / (float)random.max() * (maxvalue-minvalue) + minvalue;
    }
}
STATIC MT19937 WeightRandomizer::randomize( float *values, int numValues, float minvalue, float maxvalue ) {
    MT19937 random;
    random.seed(1); // so always gives same output
    randomize( random, values, numValues, minvalue, maxvalue );
    return random;
}
STATIC void WeightRandomizer::randomizeInts( float *values, int numValues, int minvalue, int maxvalue ) {
    MT19937 random;
    random.seed(1); // so always gives same output
    for( int i = 0; i < numValues; i++ ) {
        values[i] = ( random() % (maxvalue-minvalue) ) + minvalue;
    }
}
STATIC void WeightRandomizer::randomizeInts( int *values, int numValues, int minvalue, int maxvalue  ) {
    randomizeInts(1, values, numValues, minvalue, maxvalue );
//    MT19937 random;
//    random.seed(0); // so always gives same output
//    for( int i = 0; i < numValues; i++ ) {
//        values[i] = ( random() % (maxvalue-minvalue) ) + minvalue;
//    }
}
STATIC void WeightRandomizer::randomizeInts( int seed, int *values, int numValues, int minvalue, int maxvalue  ) {
    MT19937 random;
    if(seed == 0) {throw std::runtime_error("seed should not be zero"); } // for windows
    random.seed((unsigned long)seed); // so always gives same output
    for( int i = 0; i < numValues; i++ ) {
        values[i] = ( random() % (maxvalue-minvalue) ) + minvalue;
    }
}
STATIC void WeightRandomizer::randomizeInts( unsigned char *values, int numValues, int minvalue, int maxvalue  ) {
    randomizeInts(1, values, numValues, minvalue, maxvalue );
//    MT19937 random;
//    random.seed(0); // so always gives same output
//    for( int i = 0; i < numValues; i++ ) {
//        values[i] = ( random() % (maxvalue-minvalue) ) + minvalue;
//    }
}
STATIC void WeightRandomizer::randomizeInts( int seed, unsigned char *values, int numValues, int minvalue, int maxvalue  ) {
    MT19937 random;
    if(seed == 0) {throw std::runtime_error("seed should not be zero"); } // for windows
    random.seed((unsigned long)seed); // so always gives same output
//    random.seed(0); // so always gives same output
    for( int i = 0; i < numValues; i++ ) {
        values[i] = ( random() % (maxvalue-minvalue) ) + minvalue;
    }
}
STATIC MT19937 WeightRandomizer::randomize( ClConvolve::vfloat &values, float minvalue, float maxvalue ) {
    MT19937 random;
    random.seed(1); // so always gives same output
    randomize( random, values.begin(), values.end(), minvalue, maxvalue );
    return random;
}
template< typename It > STATIC MT19937 WeightRandomizer::randomize( It begin, It end, float minvalue, float maxvalue ) {
    MT19937 random;
    random.seed(1); // so always gives same output
    randomize( random, begin, end, minvalue, maxvalue );
    return random;
}
template< typename It > STATIC void WeightRandomizer::randomize( MT19937 &random, It begin, It end, float minvalue, float maxvalue ) {
    for( It it = begin; it != end; it++ ) {
        *it= random() / (float)random.max() * (maxvalue-minvalue) + minvalue;
    }
}
template< typename It > STATIC MT19937 WeightRandomizer::randomizeInts( It begin, It end, int minValue, int maxValue  ) {
    MT19937 random;
    random.seed(1); // so always gives same output
    randomizeInts( random, begin, end, minValue, maxValue );
    return random;
}
template< typename It > STATIC void WeightRandomizer::randomizeInts( MT19937 &random, It begin, It end, int minValue, int maxValue  ) {
    for( It it = begin; it != end; it++ ) {
        *it = ( random() % (maxValue-minValue) ) + minValue;
    }
}


template MT19937 WeightRandomizer::randomize< vfloat::iterator >( vfloat::iterator begin, vfloat::iterator end, float minvalue, float maxvalue );
template MT19937 WeightRandomizer::randomizeInts< vint::iterator >( vint::iterator begin, vint::iterator end, int minvalue, int maxvalue );

