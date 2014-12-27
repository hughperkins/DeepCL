// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class LogicalDataCreator {
public:
    int index = 0;
    float *data;
    int *labels;
    float *expectedResults;
    int N;
    const int numInputPlanes;
    const int numOutputPlanes;
    const int boardSize;
    LogicalDataCreator() :
            data(new float[8]),
            labels(new int[4]),
            expectedResults(new float[8]),
            index(0),
            numInputPlanes(2),
            numOutputPlanes(2),
            boardSize(1) {
    }
    ~LogicalDataCreator() {
        delete[] data;
        delete[] labels;
        delete[] expectedResults;
    }
    void set( bool one, bool two, bool result ) {
        data[ index * 2 ] = one ? 0.5 : -0.5;
        data[ index * 2 + 1 ] = two ? 0.5 : -0.5;
        labels[index] = result ? 1 : 0;
        expectedResults[index*2] = result ? -1 : +1;
        expectedResults[index*2+1] = result ? +1 : -1;
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


