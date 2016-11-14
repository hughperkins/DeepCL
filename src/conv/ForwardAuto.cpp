// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "conv/ForwardAuto.h"
#include "util/stringhelper.h"
#include "util/StatefulTimer.h"
#include "util/Timer.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

ForwardAuto::ForwardAuto(EasyCL *cl, LayerDimensions dim) :
        Forward(cl, dim),
        milliseconds(0),
        valid(0),
        chosenIndex(-1),
        instances(0)
         {
    num = Forward::getNumImplementations();
    milliseconds = new int[ num];
    valid = new bool[ num ];
    instances = new Forward *[ num ];
    for(int i = 0; i < num; i++) {
        instances[i] = 0;
        valid[i] = false;
        milliseconds[i] = -1;
    }
    nextIndex = 0;
}
VIRTUAL ForwardAuto::~ForwardAuto() {
    for(int i = 0; i < num; i++) {
        if(instances[i] != 0) {
            delete instances[i];
        }
    }
}
VIRTUAL void ForwardAuto::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, 
        CLWrapper *biasWrapper, CLWrapper *outputWrapper) {
//    Forward *instance = 0;
//    cout << "ForwardAuto::forward" << endl;
    while(chosenIndex == -1 && nextIndex < num) {
        int thisIndex = nextIndex;
        nextIndex++;
        cout << "forward try kernel " << thisIndex << endl;
        if(Forward::plausiblyOptimal(thisIndex, batchSize, dim)) {
            Forward *candidate = 0;
            try {
                candidate = Forward::instanceSpecific(thisIndex, cl, dim);
                instances[thisIndex] = candidate;
                valid[thisIndex] = true;
                cout << "   ... seems valid" << endl;
            } catch(runtime_error &e) {
                cout << StatefulTimer::instance()->prefix << "ForwardAuto: kernel " << thisIndex << ": this instance cant be used: " << e.what() << endl;
                valid[thisIndex] = false;
            }
            if(valid[thisIndex]) {
                Timer timer;
                try {
                    candidate->forward(batchSize, dataWrapper, weightsWrapper, biasWrapper, outputWrapper);
                    milliseconds[thisIndex] = (int)timer.lap();
                    cout << StatefulTimer::instance()->prefix << "ForwardAuto: kernel " << thisIndex << " " << milliseconds[thisIndex] << "ms" << endl;
                    if (milliseconds[thisIndex] == 0) { //we can't get better time, use this instance
                        cout << "   forward layer selected kernel with zero time" << thisIndex << endl;
                        this->chosenIndex = thisIndex;
                    }
                    return;
                } catch(runtime_error &e) {
                    cout << StatefulTimer::instance()->prefix << "ForwardAuto: kernel " << thisIndex << " this instance cant be used: " << e.what() << endl;
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
//        cout << StatefulTimer::instance()->prefix + "ForwardAuto::forward choosing best instance:" << endl;
        int bestIndex = -1;
        int bestTime = 0;
        for(int i = 0; i < num; i++) {
            if(!valid[i]) {
                cout << "   forward kernel " << i << ": cannot be used" << endl;
                continue;
            }
            cout << "   forward kernel " << i << " time: " << milliseconds[i] << "ms" << endl;
            if(bestIndex == -1) {
                bestIndex = i;
                bestTime = milliseconds[i];
                continue;
            }
            if(milliseconds[i] < bestTime) {
                bestTime = milliseconds[i];
                bestIndex = i;
            }
        }
        if(bestIndex != -1) {
            cout << "   forward layer selected kernel " << bestIndex << endl;
            this->chosenIndex = bestIndex;
        } else {
            throw runtime_error(StatefulTimer::instance()->prefix + "No valid forward implementations found");
        }
    }
//    cout << "ForwardAuto::forward using instance index: " << chosenIndex << endl;
    instances[chosenIndex]->forward(batchSize, dataWrapper, weightsWrapper, biasWrapper, outputWrapper);
}

