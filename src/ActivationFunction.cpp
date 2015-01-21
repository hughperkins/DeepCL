#include <iostream>

#include "ActivationFunction.h"

using namespace std;

ActivationFunction *ActivationFunction::fromName( std::string name ) {
    if( name == "tanh" ) {
        return new TanhActivation();
    } else if( name == "sigmoid" ) {
        return new SigmoidActivation();
    } else if( name == "linear" ) {
        return new LinearActivation();
    } else if( name == "relu" ) {
        return new ReluActivation();
    } else {
        throw std::runtime_error("activation " + name + " not known");
    }
}

ostream &operator<<( ostream &os, LinearActivation &act ) {
    os << "LinearActivation{}";
    return os;
}

ostream &operator<<( ostream &os, TanhActivation &act ) {
    os << "TanhActivation{}";
    return os;
}

ostream &operator<<( ostream &os, ReluActivation &act ) {
    os << "ReluActivation{}";
    return os;
}

ostream &operator<<( ostream &os, SigmoidActivation &act ) {
    os << "SigmoidActivation{}";
    return os;
}

