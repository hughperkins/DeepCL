// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>
#include <random>

#include "Scenario.h"

#include "ClConvolveDllExport.h"

class NeuralNet;

class Experience {
public:
    float *before;
    int action;
    float reward;
    bool isEndState;
    float *after;
};

class ClConvolve_EXPORT QLearner {
public:
    // following 4 parameters are user-configurable:
    float lambda; // means: how far into the future do we look? (any number from 0.0 to 1.0 is possible)
    int maxSamples;  // how many samples from history do we revise after each action? (default: 32)
    float epsilon; // probability of exploring, instead of exploiting, 0.0 to 1.0 ok
    float learningRate; // learning rate for the neuralnet; depends on what is appropriate for your particular
                        // network design

    QLearner( Scenario *scenario, NeuralNet *net );
    void run();  // main entry point
    virtual ~QLearner();

    void learnFromPast(); // internal method; probably not useful to user, but who knows, so leaving it 
                          // public :-)

protected:
    int size;
    int planes;
    int numActions;

    float *perception;

    std::mt19937 myrand;

    std::vector< Experience * > history;
    Scenario *scenario; // NOT belong to us, dont delete
    NeuralNet *net; // NOT belong to us, dont delete
};

