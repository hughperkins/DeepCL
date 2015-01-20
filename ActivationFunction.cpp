#include <iostream>

#include "ActivationFunction.h"

using namespace std;

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

