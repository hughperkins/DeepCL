// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <random>

#if (_MSC_VER == 1500) // visual studio 2008
#define MSVC2008
#include <ctime>
#else
#include <chrono>
#endif

class MyRandom {
    #ifdef MSVC2008
    std::tr1::mt19937 random;
    #else
    std::mt19937 random;
    #endif
    MyRandom() {
        #ifdef MSVC2008
        time_t thistime;
        time(&thistime);
        int time = thistime;
        #else
        std::chrono::time_point<std::chrono::high_resolution_clock> thistime = std::chrono::high_resolution_clock::now();
        int time = static_cast<int>( std::chrono::duration_cast<std::chrono::milliseconds> ( thistime.time_since_epoch() ).count() );
        #endif
        srand(time);
        int seed = ( rand() << 8 ) + rand();
        //MPI_Bcast( &seed, 1, MPI_INT, 0, MPI_COMM_WORLD );
        random.seed( seed );
    }
public:
    static MyRandom *instance() {
        static MyRandom *thisinstance = new MyRandom();
        return thisinstance;
    }
    static float uniform() {
        #ifdef MSVC2008
        return ( instance()->random() % 10000001 ) / 10000000.0;
        #else
        float maxrand = (float)std::mt19937::max();
        return instance()->random() / maxrand;
        #endif
    }
    static int uniformInt( int minvalue, int maxvalue ) {
        return ( instance()->random() % ( maxvalue - minvalue + 1 ) ) + minvalue;
    }
};

