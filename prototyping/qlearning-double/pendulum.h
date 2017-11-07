#ifndef PENDULUM_H
#define PENDULUM_H

#include <vector>
#include <easycl/EasyCL.h>
#include <deepcl/DeepCL.h>
#include <deepcl/qlearning/Scenario.h>
#include "PendulumEnvProxy.h"
#include "qlearning/array_helper.h"

class Pendulum : gym::pendulum::env_proxy,
        DBus::IntrospectableProxy,
        DBus::ObjectProxy,
        public Scenario
{    
    static const int numPlanes = 3;
    static const int numActions = 15;

    EasyCL *cl;
    NeuralNet *net;
    SGD *sgd;

    bool done = false;
    std::vector<double> ob;

public:
    Pendulum(DBus::Connection& connection);
    ~Pendulum();
    Trainer* getTrainer(){return sgd;}
    NeuralNet* getNeuralNet(){return net;}

    int getPerceptionSize(){return 1;}
    int getPerceptionPlanes(){return numPlanes;}
    void getPerception(float *perception);
    void reset();
    int getNumActions(){return numActions;}
    float act(int index);  // returns reward
    bool hasFinished(){return done;}
};

#endif // PENDULUM_H
