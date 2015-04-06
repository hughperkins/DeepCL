#pragma once

#include <string>
#include <stdexcept>
#include <iostream>

extern int exceptionRaised;
extern std::string exceptionMessage;
void raiseException( std::string message );
void checkException( int *wasRaised, std::string *message );

#include "NetLearner.h"

// we need this, so we can catch the c++ exception, and raise
// it in our altenrative way, all without needing to use the gil
// (which I *think* adding 'except +' requires?)
template<typename T>
class CyNetLearner : public NetLearner<T> {
public:
    CyNetLearner(NeuralNet *neuralNet ) :
        NetLearner<T>( neuralNet ) {
    }
    void learn( float learningRate ) {
        try {
            NetLearner<T>::learn(learningRate);
        } catch( std::runtime_error &e ) {
            std::cout << e.what() << std::endl;
            raiseException( e.what() );
        }
    }
};

