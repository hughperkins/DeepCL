// Copyright Hugh Perkins 2014,2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <algorithm>
#include <stdexcept>

#include "PropagateAuto.h"
#include "stringhelper.h"
#include "PropagateCpu.h"
#include "Propagate1.h"
#include "Propagate2.h"
#include "Propagate3.h"
#include "Propagate4.h"
#include "PropagateFc.h"
#include "PropagateByInputPlane.h"
#include "PropagateExperimental.h"
#include "StatefulTimer.h"
#include "Timer.h"

using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

PropagateAuto::PropagateAuto( OpenCLHelper *cl, LayerDimensions dim, ActivationFunction const*fn ) :
//        dim( layerDimensions ),
//        cl( cl ),
//        fn( fn ),
        Propagate( cl, dim, fn ),
        milliseconds( 0 ),
        valid( 0 ),
        chosenIndex( -1 ),
        instances( 0 )
         {
    num = Propagate::getNumImplementations();
    milliseconds = new int[ num];
    valid = new bool[ num ];
    instances = new Propagate *[ num ];
    for( int i = 0; i < num; i++ ) {
        instances[i] = 0;
        valid[i] = false;
        milliseconds[i] = -1;
    }
    nextIndex = 0;
}
VIRTUAL PropagateAuto::~PropagateAuto() {
    for( int i = 0; i < num; i++ ) {
        if( instances[i] != 0 ) {
            delete instances[i];
        }
    }
}
VIRTUAL void PropagateAuto::propagate( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, 
        CLWrapper *biasWeightsWrapper, CLWrapper *resultsWrapper ) {
//    Propagate *instance = 0;
//    cout << "PropagateAuto::propagate" << endl;
    while( chosenIndex == -1 && nextIndex < num ) {
        int thisIndex = nextIndex;
        nextIndex++;
        if( Propagate::plausiblyOptimal( thisIndex, batchSize, dim, fn ) ) {
            Propagate *candidate = 0;
            try {
                candidate = Propagate::instanceSpecific( thisIndex, cl, dim, fn );
                instances[thisIndex] = candidate;
                valid[thisIndex] = true;
            } catch( runtime_error &e ) {
                cout << StatefulTimer::instance()->prefix << "PropagateAuto: instance " << thisIndex << ": this instance cant be used: " << e.what() << endl;
                valid[thisIndex] = false;
            }
            if( valid[thisIndex] ) {
                Timer timer;
                try {
                    candidate->propagate( batchSize, dataWrapper, weightsWrapper, biasWeightsWrapper, resultsWrapper );
                    milliseconds[thisIndex] = timer.lap();
//                    cout << StatefulTimer::instance()->prefix << "PropagateAuto: instance " << thisIndex << " " << milliseconds[thisIndex] << "ms" << endl;
                    return;
                } catch( runtime_error &e ) {
                    cout << StatefulTimer::instance()->prefix << "PropagateAuto: instance " << thisIndex << " this instance cant be used: " << e.what() << endl;
                    valid[thisIndex] = false;
                    delete instances[thisIndex];
                    instances[thisIndex] = 0;
                }
            }
        }
    }
    if( chosenIndex == -1 ) {
//        cout << StatefulTimer::instance()->prefix + "PropagateAuto::propagate choosing best instance, based on measured times..." << endl;
        int bestIndex = -1;
        int bestTime = 0;
        for( int i = 0; i < num; i++ ) {
            if( !valid[i] ) {
                continue;
            }
            if( bestIndex == -1 ) {
                bestIndex = i;
                bestTime = milliseconds[i];
                continue;
            }
            if( milliseconds[i] < bestTime ) {
                bestTime = milliseconds[i];
                bestIndex = i;
            }
        }
        if( bestIndex != -1 ) {
            cout << StatefulTimer::instance()->prefix << "PropagateAuto: selected index: " << bestIndex << endl;
            this->chosenIndex = bestIndex;
        } else {
            throw runtime_error(StatefulTimer::instance()->prefix + "No valid propagate implementations found" );
        }
    }
//    cout << "PropagateAuto::propagate using instance index: " << chosenIndex << endl;
    instances[chosenIndex]->propagate( batchSize, dataWrapper, weightsWrapper, biasWeightsWrapper, resultsWrapper );
}

