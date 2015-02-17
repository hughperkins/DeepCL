// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <random>

#include "test/WeightRandomizer.h"
#include "ClConvolve_typedefs.h"

using namespace std;
using namespace ClConvolve;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC void WeightRandomizer::randomize( int seed, float *values, int numValues, float minvalue, float maxvalue ) {
    std::mt19937 random;
    random.seed(seed); // so always gives same results
    randomize( random, values, numValues, minvalue, maxvalue );
}
STATIC void WeightRandomizer::randomize( std::mt19937 &random, float *values, int numValues, float minvalue, float maxvalue ) {
    for( int i = 0; i < numValues; i++ ) {
        values[i] = random() / (float)std::mt19937::max() * (maxvalue-minvalue) - maxvalue;
    }
}
STATIC std::mt19937 WeightRandomizer::randomize( float *values, int numValues, float minvalue, float maxvalue ) {
    std::mt19937 random;
    random.seed(0); // so always gives same results
    randomize( random, values, numValues, minvalue, maxvalue );
    return random;
}
STATIC void WeightRandomizer::randomizeInts( float *values, int numValues, int minvalue, int maxvalue ) {
    std::mt19937 random;
    random.seed(0); // so always gives same results
    for( int i = 0; i < numValues; i++ ) {
        values[i] = ( random() % (maxvalue-minvalue) ) + minvalue;
    }
}
STATIC void WeightRandomizer::randomizeInts( int *values, int numValues, int minvalue, int maxvalue  ) {
    std::mt19937 random;
    random.seed(0); // so always gives same results
    for( int i = 0; i < numValues; i++ ) {
        values[i] = ( random() % (maxvalue-minvalue) ) + minvalue;
    }
}
STATIC std::mt19937 WeightRandomizer::randomize( ClConvolve::vfloat &values, float minvalue, float maxvalue ) {
    std::mt19937 random;
    random.seed(0); // so always gives same results
    randomize( random, values.begin(), values.end(), minvalue, maxvalue );
    return random;
}
template< typename It > STATIC std::mt19937 WeightRandomizer::randomize( It begin, It end, float minvalue, float maxvalue ) {
    std::mt19937 random;
    random.seed(0); // so always gives same results
    randomize( random, begin, end, minvalue, maxvalue );
    return random;
}
template< typename It > STATIC void WeightRandomizer::randomize( std::mt19937 &random, It begin, It end, float minvalue, float maxvalue ) {
    for( It it = begin; it != end; it++ ) {
        *it= random() / (float)std::mt19937::max() * (maxvalue-minvalue) - maxvalue;
    }
}
template< typename It > STATIC std::mt19937 WeightRandomizer::randomizeInts( It begin, It end, int minValue, int maxValue  ) {
    std::mt19937 random;
    random.seed(0); // so always gives same results
    randomizeInts( random, begin, end, minValue, maxValue );
    return random;
}
template< typename It > STATIC void WeightRandomizer::randomizeInts( std::mt19937 &random, It begin, It end, int minValue, int maxValue  ) {
    for( It it = begin; it != end; it++ ) {
        *it = ( random() % (maxValue-minValue) ) + minValue;
    }
}


template std::mt19937 WeightRandomizer::randomize< vfloat::iterator >( vfloat::iterator begin, vfloat::iterator end, float minvalue, float maxvalue );
template std::mt19937 WeightRandomizer::randomizeInts< vint::iterator >( vint::iterator begin, vint::iterator end, int minvalue, int maxvalue );

