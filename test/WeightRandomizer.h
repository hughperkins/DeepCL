// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <random>

#include "ClConvolve_typedefs.h"

#define VIRTUAL virtual
#define STATIC static

class WeightRandomizer {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC void randomize( int seed, float *values, int numValues, float minvalue, float maxvalue );
    STATIC void randomize( std::mt19937 &random, float *values, int numValues, float minvalue, float maxvalue );
    STATIC std::mt19937 randomize( float *values, int numValues, float minvalue, float maxvalue );
    STATIC void randomizeInts( float *values, int numValues, int minvalue, int maxvalue );
    STATIC void randomizeInts( int *values, int numValues, int minvalue, int maxvalue  );
    STATIC std::mt19937 randomize( ClConvolve::vfloat &values, float minvalue, float maxvalue );
    template< typename It > STATIC std::mt19937 randomize( It begin, It end, float minvalue, float maxvalue );
    template< typename It > STATIC void randomize( std::mt19937 &random, It begin, It end, float minvalue, float maxvalue );
    template< typename It > STATIC std::mt19937 randomizeInts( It begin, It end, int minValue, int maxValue  );
    template< typename It > STATIC void randomizeInts( std::mt19937 &random, It begin, It end, int minValue, int maxValue  );

    // [[[end]]]
};

