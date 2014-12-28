// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>
#include <stdexcept>

class ActivationFunction {
public:
    virtual float calc( float value ) const { throw std::runtime_error("calc not implemented"); };
    virtual float calcDerivative( float output ) const { throw std::runtime_error("calcDerivative not implemented"); };
    virtual std::string getKernelFunction() const { throw std::runtime_error("getKernelFunction not implemented"); };
    virtual float getFalse() const {  throw std::runtime_error("getFalse not implemented"); } 
    virtual float getTrue() const {  throw std::runtime_error("getTrue not implemented"); } 
};

class TanhActivation : public ActivationFunction {
public:
    virtual std::string getKernelFunction() const {
        return "byelement_tanh";
    }
    virtual float calc( float value ) const {
        return tanh( value );
    }
    virtual float calcDerivative( float output ) const {
        return 1 - output * output;
    }
    virtual float getTrue() const {
        return 0.5;
    }
    virtual float getFalse() const {
        return -0.5;
    }
};

class LinearActivation : public ActivationFunction {
public:
    virtual std::string getKernelFunction() const {
        return "";
    }
    virtual float calc( float value ) const {
        return value;
    }
    virtual float calcDerivative( float output ) const {
        return 1;
    }
    virtual float getTrue() const {
        return 0.5;
    }
    virtual float getFalse() const {
        return -0.5;
    }
};

class ReluActivation : public ActivationFunction {
public:
    virtual std::string getKernelFunction() const {
        return "byelement_relu";
    }
    virtual float calc( float value ) const {
        return value > 0 ? value : 0;
    }
    virtual float calcDerivative( float output ) const {
        return output > 0 ? 1 : 0;
    }
    virtual float getTrue() const {
        return 0.8;
    }
    virtual float getFalse() const {
        return 0.2;
    }
};



