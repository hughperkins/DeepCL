// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "conv/BackpropWeightsAuto.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "util/Timer.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackpropWeightsAuto::BackpropWeightsAuto(EasyCL *cl, LayerDimensions dim) :
        BackpropWeights(cl, dim),
        microseconds(0),
        valid(0),
        chosenIndex(-1),
        instances(0)
         {
    num = BackpropWeights::getNumImplementations();
    microseconds = new int[num];
    numTries = new int[num];
    valid = new bool[ num ];
    instances = new BackpropWeights *[ num ];
    for(int i = 0; i < num; i++) {
        instances[i] = 0;
        valid[i] = false;
        numTries[i] = 0;
        microseconds[i] = -1;
    }
    currentIndex = 0;
}
VIRTUAL BackpropWeightsAuto::~BackpropWeightsAuto() {
    for(int i = 0; i < num; i++) {
        if(instances[i] != 0) {
            delete instances[i];
        }
    }

    delete[] microseconds;
    delete[] numTries;
    delete[] valid;
    delete[] instances;
}
VIRTUAL void BackpropWeightsAuto::calcGradWeights(
        int batchSize, CLWrapper *inputDataWrapper, CLWrapper *gradOutput, CLWrapper *weightsWrapper,
        CLWrapper *gradInput) {
    while(chosenIndex == -1 && currentIndex < num) {
        BackpropWeights *candidate = instances[currentIndex];
        if(candidate == 0) {
            cout << "calcGradWeights try kernel " << currentIndex << endl;
            if(!BackpropWeights::plausiblyOptimal(currentIndex, batchSize, dim)) {
                cout << "  ... not plausibly optimal, skipping" << endl;
                valid[currentIndex] = false;
                currentIndex++;
                continue;
            }
            try {
                candidate = BackpropWeights::instanceSpecific(currentIndex, cl, dim);
                instances[currentIndex] = candidate;
                valid[currentIndex] = true;
                cout << "   ... seems valid" << endl;
            } catch(runtime_error &e) {
                cout << "   ... not valid" << endl;
                cout << StatefulTimer::instance()->prefix << "BackpropWeightsAuto: kernel " << currentIndex << ": this instance cant be used: " << e.what() << endl;
                valid[currentIndex] = false;
                currentIndex++;
                continue;
            }
        }
        Timer timer;
        try {
            candidate->calcGradWeights(batchSize, inputDataWrapper, gradOutput, weightsWrapper, gradInput);
            microseconds[currentIndex] = (int)timer.elapsedMicroseconds();
            cout << "  try " << (numTries[currentIndex]+ 1) << " of kernel " << currentIndex << " time " << microseconds[currentIndex] << endl;
            cout << StatefulTimer::instance()->prefix << "BackpropWeightsAuto: kernel " << currentIndex << " " << microseconds[currentIndex] << "us" << endl;
            numTries[currentIndex]++;
            if(numTries[currentIndex] >= 3) {  // we already tried this kernel 3 times, try next kernel
                 currentIndex++;
            }
            return;
        } catch(runtime_error &e) {
            cout << StatefulTimer::instance()->prefix << "BackpropWeightsAuto: kernel " << currentIndex << " this instance cant be used: " << e.what() << endl;
            valid[currentIndex] = false;
            delete instances[currentIndex];
            instances[currentIndex] = 0;
            currentIndex++;
            continue;
        }
    }
    if(chosenIndex == -1) {
//        cout << StatefulTimer::instance()->prefix + "BackpropWeightsAuto::calcGradWeights choosing best instance:" << endl;
        int bestIndex = -1;
        int bestTime = 0;
        for(int i = 0; i < num; i++) {
            if(!valid[i]) {
                cout << "   calcGradWeights kernel " << i << ": cannot be used" << endl;
                continue;
            }
            cout << "   calcGradWeights kernel " << i << " time: " << microseconds[i] << "us" << endl;
            if(bestIndex == -1) {
                bestIndex = i;
                bestTime = microseconds[i];
                continue;
            }
            if(microseconds[i] < bestTime) {
                bestTime = microseconds[i];
                bestIndex = i;
            }
        }
        if(bestIndex != -1) {
            cout << "   calcGradWeights layer selected kernel " << bestIndex << endl;
            this->chosenIndex = bestIndex;
        } else {
            throw runtime_error(StatefulTimer::instance()->prefix + "No valid calcGradWeights implementations found");
        }
    }
//    cout << "BackpropWeightsAuto::calcGradWeights using instance index: " << chosenIndex << endl;
    instances[chosenIndex]->calcGradWeights(batchSize, inputDataWrapper, gradOutput, weightsWrapper, gradInput);
}

