// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <random>

#include "mt19937defs.h"

//#if (_MSC_VER == 1500 || _MSC_VER == 1600  )
#ifdef _MSC_VER // make consistent across all msvc versions, so dont have to retest on different msvc versions...
#define TR1RANDOM
#endif

//#if (_MSC_VER == 1500 || _MSC_VER == 1600 ) // visual studio 2008
#ifdef _MSC_VER // make consistent across all msvc versions, so dont have to retest on different msvc versions...
#define NOCHRONO
#include <ctime>
#else
#include <chrono>
#endif

class MyRandom {
//    #ifdef TR1RANDOM
    MT19937 random;
//    #else
//    std::mt19937 random;
//    #endif
    MyRandom() {
        #ifdef NOCHRONO
        time_t thistime;
        time(&thistime);
        int time = (int)thistime;
        #else
        std::chrono::time_point<std::chrono::high_resolution_clock> thistime = std::chrono::high_resolution_clock::now();
        int time = static_cast<int>( std::chrono::duration_cast<std::chrono::milliseconds> ( thistime.time_since_epoch() ).count() );
        #endif
        srand(time);
        unsigned long seed = ( rand() << 8 ) + rand();
        //MPI_Bcast( &seed, 1, MPI_INT, 0, MPI_COMM_WORLD );
        random.seed( seed );
    }
public:
    static MyRandom *instance() {
        static MyRandom *thisinstance = new MyRandom();
        return thisinstance;
    }
    float _uniform() {
//        float maxrand = (float)random.max();
        return random() / (float)random.max();
    }
    static float uniform() {
//        #ifdef TR1RANDOM
//        return ( instance()->random() % 10000001 ) / 10000000.0f;
//        #else
//        float maxrand = (float)instance()->random.max();
//        return instance()->random() / maxrand;
        return instance()->_uniform();
//        #endif
    }
    static int uniformInt( int minvalue, int maxvalue ) {
        return ( instance()->random() % ( maxvalue - minvalue + 1 ) ) + minvalue;
    }
};

