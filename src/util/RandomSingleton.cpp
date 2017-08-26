// Copyright Hugh Perkins 2014, 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "util/RandomSingleton.h"

#undef STATIC
#define STATIC
#undef VIRTUAL
#define VIRTUAL

PUBLIC RandomSingleton::RandomSingleton() {
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
        time = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds> (thistime.time_since_epoch()).count());
    }
    #endif
    srand(time);
    unsigned long seed = (rand() << 8) + rand();
    myrandom.seed(seed);
}
PUBLIC STATIC RandomSingleton *RandomSingleton::instance() {
    static RandomSingleton *thisinstance = new RandomSingleton();
    return thisinstance; // assume single-threaded, which... we are :-)
}
//    void testingonly_setInstance(RandomSingleton *testInstance) {
//        _instance = testinstance;
//    }
PUBLIC VIRTUAL float RandomSingleton::_uniform() {
    return myrandom() / (float)myrandom.max();
}
PUBLIC STATIC void RandomSingleton::seed(unsigned long seed) {
    instance()->myrandom.seed(seed);
}
PUBLIC STATIC float RandomSingleton::uniform() {
    return instance()->_uniform();
}
PUBLIC STATIC int RandomSingleton::uniformInt(int minValueInclusive, int maxValueInclusive) {
    return (instance()->myrandom() % 
        (maxValueInclusive - minValueInclusive + 1) )
     + minValueInclusive;
}

