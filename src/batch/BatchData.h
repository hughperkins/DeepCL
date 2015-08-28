// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class Trainable;

class OutputData {
public:
    OutputData() {
    }
    virtual ~OutputData() {
    }
    virtual OutputData *slice(int start) = 0;
//    static LabeledData *fromLabels(int *labels) {
//        return new LabeledData(labels);
//    }
//    static ExpectedData *fromExpected(int outputCubeSize, float *expected) {
//        return new ExpectedData(outputCubeSize, expected);
//    }
};
class LabeledData : public OutputData {
public:
    int const*labels; // NOT owned by us, dont delete
    LabeledData(int const*labels) {
        this->labels = labels;
    }
    LabeledData(Trainable *net, int const*labels);
    static LabeledData *instance(Trainable *net, int const*labels);
    LabeledData *slice(int start) {
        LabeledData *child = new LabeledData(labels + start);
        return child;
    }
};
class ExpectedData : public OutputData {
public:
    int outputCubeSize;
    float const*expected; // NOT owned by us, dont delete

    ExpectedData(int outputCubeSize, float const*expected) {
        this->outputCubeSize = outputCubeSize;
        this->expected = expected;
    }
    ExpectedData(Trainable *net, float const*expected);
    static ExpectedData *instance(Trainable *net, float const*expected);
    ExpectedData *slice(int start) {
        ExpectedData *child = new ExpectedData(outputCubeSize, expected + start * outputCubeSize);
        return child;
    }
};
class InputData {
public:
    int inputCubeSize;
    float const*inputs; // NOT owned by us, dont delete
    InputData(int inputCubeSize, float const*inputs) {
        this->inputCubeSize = inputCubeSize;
        this->inputs = inputs;
    }
    static InputData *instance(Trainable *net, float const*inputs);
    InputData *slice(int start) {
        InputData *child = new InputData(inputCubeSize, inputs + start * inputCubeSize);
        return child;
    }
};
