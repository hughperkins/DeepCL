#pragma once

#include <string>
#include <stdexcept>

class ActivationFunction {
public:
    virtual float calc( float value ) const { throw std::runtime_error("calc not implemented"); };
    virtual float calcDerivative( float output ) const { throw std::runtime_error("calcDerivative not implemented"); };
    virtual std::string getKernelFunction() const { throw std::runtime_error("getKernelFunction not implemented"); };
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
};



