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

#include "DeepCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class DeepCL_EXPORT Op1 {
public:
    virtual std::string getOperationString() = 0;
    virtual std::string getName() = 0;
};
class DeepCL_EXPORT Op1Inv : public Op1 {
    std::string getOperationString() {
        return "1.0f / val_one";
    }
    std::string getName(){ return "Op1_Inv"; }
};
class DeepCL_EXPORT Op1Sqrt : public Op1 {
    std::string getOperationString() {
        return "native_sqrt(val_one)";
    }
    std::string getName(){ return "Op1_Sqrt"; }
};
class DeepCL_EXPORT Op1Squared : public Op1 {
    std::string getOperationString() {
        return "val_one * val_one";
    }
    std::string getName(){ return "Op1_Squared"; }
};

class DeepCL_EXPORT Op2 {
public:
    virtual std::string getOperationString() = 0;
    virtual std::string getName() = 0;
};
class DeepCL_EXPORT Op2Equal : public Op2 {
    std::string getOperationString() {
        return "val_two";
    }
    std::string getName(){ return "Op2_Equal"; }
};
class DeepCL_EXPORT Op2Add : public Op2 {
public:
    std::string getOperationString() {
        return "val_one + val_two";
    }
    std::string getName(){ return "Op2_Add"; }
};
class DeepCL_EXPORT Op2Mul : public Op2 {
public:
    std::string getOperationString() {
        return "val_one * val_two";
    }
    std::string getName(){ return "Op2_Mul"; }
};
class DeepCL_EXPORT Op2Sub : public Op2 {
public:
    std::string getOperationString() {
        return "val_one - val_two";
    }
    std::string getName(){ return "Op2_Sub"; }
};
class DeepCL_EXPORT Op2Div : public Op2 {
public:
    std::string getOperationString() {
        return "val_one / val_two";
    }
    std::string getName(){ return "Op2_Div"; }
};


// use to update one buffer by adding another buffer, in-element
// not thread-safe
class DeepCL_EXPORT GpuOp {
public:
    EasyCL *cl; // NOT belong to us, dont delete
    CLKernel *kernel;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL void apply2_inplace(int N, CLWrapper*destinationWrapper, float scalar, Op2 *op);
    VIRTUAL void apply2_inplace(int N, CLWrapper*destinationWrapper, CLWrapper *deltaWrapper, Op2 *op);
    VIRTUAL void apply2_outofplace(int N, CLWrapper*destinationWrapper, CLWrapper*one, CLWrapper *two, Op2 *op);
    VIRTUAL void apply1_inplace(int N, CLWrapper*destinationWrapper, Op1 *op);
    VIRTUAL void apply1_outofplace(int N, CLWrapper*destinationWrapper, CLWrapper*one, Op1 *op);
    VIRTUAL ~GpuOp();
    GpuOp(EasyCL *cl);
    void buildKernel(std::string name, Op2 *op, bool inPlace);
    void buildKernel(std::string name, Op1 *op, bool inPlace);
    void buildKernelScalar(std::string name, Op2 *op, bool inPlace);

    // [[[end]]]
};

