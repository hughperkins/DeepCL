#include <iostream>

#include "activate/ActivationFunction.h"

using namespace std;

ActivationFunction *ActivationFunction::fromName(std::string name) {
    if(name == "tanh") {
        return new TanhActivation();
    } else if(name == "scaledtanh") {
        return new ScaledTanhActivation();
    } else if(name == "sigmoid") {
        return new SigmoidActivation();
    } else if(name == "linear") {
        return new LinearActivation();
    } else if(name == "relu") {
        return new ReluActivation();
    } else if(name == "elu") {
        return new EluActivation();
    } else {
        throw std::runtime_error("activation " + name + " not known");
    }
}

ostream &operator<<(ostream &os, LinearActivation const&act) {
    os << "LinearActivation{}";
    return os;
}

ostream &operator<<(ostream &os, TanhActivation const&act) {
    os << "TanhActivation{}";
    return os;
}

ostream &operator<<(ostream &os, ScaledTanhActivation const&act) {
    os << "ScaledTanhActivation{}";
    return os;
}

ostream &operator<<(ostream &os, EluActivation const&act) {
    os << "EluActivation{}";
    return os;
}

ostream &operator<<(ostream &os, ReluActivation const&act) {
    os << "ReluActivation{}";
    return os;
}

ostream &operator<<(ostream &os, SigmoidActivation const&act) {
    os << "SigmoidActivation{}";
    return os;
}

