// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "activate/ActivationFunction.h"

class LogicalDataCreator {
public:
    int index;
    float *data;
    int *labels;
    float *expectedOutput;
    int N;
    const int numInputPlanes;
    const int numOutputPlanes;
    const int imageSize;
    ActivationFunction *fn;
    LogicalDataCreator( ActivationFunction *activationFunction = new TanhActivation() ) :
			index(0),
            data(new float[8]),
            labels(new int[4]),
            expectedOutput(new float[8]),
            numInputPlanes(2),
            numOutputPlanes(2),
            imageSize(1),
            fn( activationFunction ) {
    }
    ~LogicalDataCreator() {
        delete[] data;
        delete[] labels;
        delete[] expectedOutput;
    }
    void set( bool one, bool two, bool result ) {
        data[ index * 2 ] = one ? 0.5 : -0.5;
        data[ index * 2 + 1 ] = two ? 0.5 : -0.5;
        labels[index] = result ? 1 : 0;
        expectedOutput[index*2] = result ? fn->getFalse() : fn->getTrue();
        expectedOutput[index*2+1] = result ? fn->getTrue() : fn->getFalse();
        index++;
        N++;
    }

    void applyAndGate() {
       index = 0;
       N = 0;
       set( false, false, false );
       set( false, true, false );
       set( true, false, false );
       set( true, true, true );
    }

    void applyOrGate() {
       index = 0;
       N = 0;
       set( false, false, false );
       set( false, true, true );
       set( true, false, true );
       set( true, true, true );
    }

    void applyXorGate() {
       index = 0;
       N = 0;
       set( false, false, false );
       set( false, true, true );
       set( true, false, true );
       set( true, true, false );
    }
};


