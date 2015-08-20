// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class NeuralNet;
 
class Scenario {
public:
    virtual ~Scenario() {}
//    virtual void print() {} // optional implementation
//    virtual void printQRepresentation(NeuralNet *net) {} // optional implementation
    virtual int getPerceptionSize() = 0;
    virtual int getPerceptionPlanes() = 0;
    virtual void getPerception(float *perception) = 0;
    virtual void reset() = 0;
    virtual int getNumActions() = 0;
    virtual float act(int index) = 0;  // returns reward
    virtual bool hasFinished() = 0;
//    virtual int getWorldSize() = 0;
};

