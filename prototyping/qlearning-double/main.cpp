#include <iostream>
#include "pendulum.h"
#include "deepcl/qlearning/QLearner.h"
#include "NatureQLearner.h"
#include "DoubleQLearner.h"

int main(int argc, char *argv[])
{
    // DBus Init
    DBus::BusDispatcher dispatcher;
    DBus::default_dispatcher = &dispatcher;
    DBus::Connection bus = DBus::Connection::SessionBus();

    Pendulum *env = new Pendulum(bus);
    // Reset the gym
    env->reset();

//    QLearner qLearner( env->getTrainer(), env, env->getNeuralNet() );

//    NatureQLearner qLearner( env->getTrainer(), env, env->getNeuralNet() );

    DoubleQLearner qLearner( env->getTrainer(), env, env->getNeuralNet() );

    qLearner.run();
}
