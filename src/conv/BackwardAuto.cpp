// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "conv/BackwardAuto.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "util/Timer.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

BackwardAuto::BackwardAuto(EasyCL *cl, LayerDimensions dim) :
        Backward(cl, dim),
        microseconds(0),
        valid(0),
        chosenIndex(-1),
        instances(0)
         {
    num = Backward::getNumImplementations();
    microseconds = new int[ num];
    valid = new bool[ num ];
    instances = new Backward *[ num ];
    for(int i = 0; i < num; i++) {
        instances[i] = 0;
        valid[i] = false;
        microseconds[i] = -1;
    }
    nextIndex = 0;
}
VIRTUAL BackwardAuto::~BackwardAuto() {
    for(int i = 0; i < num; i++) {
        if(instances[i] != 0) {
            delete instances[i];
        }
    }

    delete[] microseconds;
    delete[] valid;
    delete[] instances;
}
VIRTUAL void BackwardAuto::backward(
        int batchSize, CLWrapper *inputDataWrapper, CLWrapper *gradOutput, CLWrapper *weightsWrapper,
        CLWrapper *gradInput) {
    while(chosenIndex == -1 && nextIndex < num) {
        int thisIndex = nextIndex;
        nextIndex++;
        cout << "backward try kernel " << thisIndex << endl;
        if(Backward::plausiblyOptimal(thisIndex, batchSize, dim)) {
            Backward *candidate = 0;
            try {
                candidate = Backward::instanceSpecific(thisIndex, cl, dim);
                instances[thisIndex] = candidate;
                valid[thisIndex] = true;
                cout << "   ... seems valid" << endl;
            } catch(runtime_error &e) {
                cout << StatefulTimer::instance()->prefix << "BackwardAuto: kernel " << thisIndex << ": this instance cant be used: " << e.what() << endl;
                valid[thisIndex] = false;
            }
            if(valid[thisIndex]) {
                Timer timer;
                try {
                    candidate->backward(batchSize, inputDataWrapper, gradOutput, weightsWrapper, gradInput);
                    microseconds[thisIndex] = (int)timer.lap();
                    cout << StatefulTimer::instance()->prefix << "BackwardAuto: kernel " << thisIndex << " " << microseconds[thisIndex] << "us" << endl;
                    return;
                } catch(runtime_error &e) {
                    cout << StatefulTimer::instance()->prefix << "BackwardAuto: kernel " << thisIndex << " this instance cant be used: " << e.what() << endl;
                    valid[thisIndex] = false;
                    delete instances[thisIndex];
                    instances[thisIndex] = 0;
                }
            } else {
                cout << "   ... not valid" << endl;
            }
        } else {
            cout << "  ... not plausibly optimal, skipping" << endl;
        }
    }
    if(chosenIndex == -1) {
//        cout << StatefulTimer::instance()->prefix + "BackwardAuto::backward choosing best instance:" << endl;
        int bestIndex = -1;
        int bestTime = 0;
        for(int i = 0; i < num; i++) {
            if(!valid[i]) {
                cout << "   backward kernel " << i << ": cannot be used" << endl;
                continue;
            }
            cout << "   backward kernel " << i << " time: " << microseconds[i] << "us" << endl;
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
            cout << "   backward layer selected kernel " << bestIndex << endl;
            this->chosenIndex = bestIndex;
        } else {
            throw runtime_error(StatefulTimer::instance()->prefix + "No valid backward implementations found");
        }
    }
//    cout << "BackwardAuto::backward using instance index: " << chosenIndex << endl;
    instances[chosenIndex]->backward(batchSize, inputDataWrapper, gradOutput, weightsWrapper, gradInput);
}

