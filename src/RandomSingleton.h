// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <random>

#include "mt19937defs.h"

#ifdef _MSC_VER // apply to all msvc versions, so dont have to retest on different msvc versions
#define TR1RANDOM
#endif

#ifdef _MSC_VER // apply to all msvc versions, so dont have to retest on different msvc versions
#define NOCHRONO
#include <ctime>
#else
#include <chrono>
#endif

// singleton version of mt19937, so we seed it once, based on current time
// and then keep getting values out of it, even if used from different places
// and classes
// probably not threadsafe
// constructor is public, so we can override it, for testing, if we want
class RandomSingleton {
public:
    MT19937 random;
    RandomSingleton() {
        int time = 0;
        #ifdef NOCHRONO
        {
            time_t thistime;
            ::time(&thistime);
            time = (int)thistime;
        }
        #else
        {
            std::chrono::time_point<std::chrono::high_resolution_clock> thistime = std::chrono::high_resolution_clock::now();
            time = static_cast<int>( std::chrono::duration_cast<std::chrono::milliseconds> ( thistime.time_since_epoch() ).count() );
        }
        #endif
        srand(time);
        unsigned long seed = ( rand() << 8 ) + rand();
        random.seed( seed );
    }
    static RandomSingleton *instance() {
        static RandomSingleton *thisinstance = new RandomSingleton();
        return thisinstance; // assume single-threaded, which... we are :-)
    }
//    void testingonly_setInstance( RandomSingleton *testInstance ) {
//        _instance = testinstance;
//    }
    virtual float _uniform() {
        return random() / (float)random.max();
    }
    static float uniform() {
        return instance()->_uniform();
    }
    static int uniformInt( int minValueInclusive, int maxValueInclusive ) {
        return ( instance()->random() % 
            ( maxValueInclusive - minValueInclusive + 1 ) )
         + minValueInclusive;
    }
};

