// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

class EasyCL;
class CLWrapper;
class CLKernel;

#define VIRTUAL virtual
#define STATIC static

class Op2 {
public:
    virtual std::string getOperationString() = 0;
    virtual std::string getName() = 0;
};
class Op2Add : public Op2 {
public:
    std::string getOperationString() {
        return "val_one + val_two";
    }
    std::string getName(){ return "Add"; }
};
class Op2Mul : public Op2 {
public:
    std::string getOperationString() {
        return "val_one * val_two";
    }
    std::string getName(){ return "Mul"; }
};
class Op2Sub : public Op2 {
public:
    std::string getOperationString() {
        return "val_one - val_two";
    }
    std::string getName(){ return "Sub"; }
};
class Op2Div : public Op2 {
public:
    std::string getOperationString() {
        return "val_one / val_two";
    }
    std::string getName(){ return "Div"; }
};


// use to update one buffer by adding another buffer, in-element
// not thread-safe
class GpuOp {
public:
    EasyCL *cl; // NOT belong to us, dont delete
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL void apply2_inplace( int N, CLWrapper*destinationWrapper, CLWrapper *deltaWrapper, Op2 *op );
    VIRTUAL void apply2_outofplace( int N, CLWrapper*destinationWrapper, CLWrapper*one, CLWrapper *two, Op2 *op );
    VIRTUAL ~GpuOp();
    GpuOp( EasyCL *cl );
    void buildKernel( std::string name, Op2 *op, bool inPlace );

    // [[[end]]]
};

