// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class OutputData {
public:
    OutputData() {
    }
    virtual ~OutputData() {
    }
    virtual OutputData *slice( int start ) = 0;
//    static LabeledData *fromLabels( int *labels ) {
//        return new LabeledData( labels );
//    }
//    static ExpectedData *fromExpected( int outputCubeSize, float *expected ) {
//        return new ExpectedData( outputCubeSize, expected );
//    }
};
class LabeledData : public OutputData {
public:
    int *labels; // NOT owned by us, dont delete
    LabeledData( int *labels ) {
        this->labels = labels;
    }
    LabeledData *slice( int start ) {
        LabeledData *child = new LabeledData( labels + start );
        return child;
    }
};
class ExpectedData : public OutputData {
public:
    int outputCubeSize;
    float *expected; // NOT owned by us, dont delete
    ExpectedData( int outputCubeSize, float *expected ) {
        this->outputCubeSize = outputCubeSize;
        this->expected = expected;
    }
    ExpectedData *slice( int start ) {
        ExpectedData *child = new ExpectedData( outputCubeSize, expected + start * outputCubeSize );
        return child;
    }
};
class InputData {
public:
    int inputCubeSize;
    float *inputs; // NOT owned by us, dont delete
    InputData( int inputCubeSize, float *inputs ) {
        this->inputCubeSize = inputCubeSize;
        this->inputs = inputs;
    }
    InputData *slice( int start ) {
        InputData *child = new InputData( inputCubeSize, inputs + start * inputCubeSize );
        return child;
    }
};
