// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <random>

class MyRandom {
    std::mt19937 random;
    MyRandom() {
        srand(time(0));
        int seed = rand() << 8 + rand();
        //MPI_Bcast( &seed, 1, MPI_INT, 0, MPI_COMM_WORLD );
        random.seed( seed );
    }
public:
    static MyRandom *instance() {
        static MyRandom *thisinstance = new MyRandom();
        return thisinstance;
    }
    static float uniform() {
        return instance()->random() / (float)std::mt19937::max();
    }
};

