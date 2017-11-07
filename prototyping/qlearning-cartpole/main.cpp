#include <iostream>

#include "deepcl/DeepCL.h"
#include "deepcl/qlearning/QLearner.h"
#include "cartenv.h"

#include <vector>


int main(int argc, char *argv[])
{
    // DBus Init
    DBus::BusDispatcher dispatcher;
    DBus::default_dispatcher = &dispatcher;
    DBus::Connection bus = DBus::Connection::SessionBus();

    CartEnv *env = new CartEnv(bus);
    // Reset the gym
    env->reset();

    // Init DeepCL
    EasyCL *cl = new EasyCL();
    NeuralNet *net = new NeuralNet( cl );
    SGD *sgd = SGD::instance( cl, 0.01f, 0.001f );

    const int size = env->getPerceptionSize();
    const int planes = env->getPerceptionPlanes();
    const int numActions = env->getNumActions();
    net->addLayer( InputLayerMaker::instance()->numPlanes(planes)->imageSize(size) );
    net->addLayer( ConvolutionalMaker::instance()->filterSize(size)->numFilters(planes*4)->biased()->padZeros() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->filterSize(size)->numFilters(planes*2)->biased()->padZeros() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( FullyConnectedMaker::instance()->imageSize(size)->numPlanes(size*size*planes*10)->biased() );
        net->addLayer( ActivationMaker::instance()->tanh() );
    net->addLayer( FullyConnectedMaker::instance()->imageSize(size)->numPlanes(numActions)->biased() );
    net->addLayer( SquareLossMaker::instance() );
    net->print();

    QLearner qLearner( sgd, env, net );
    qLearner.run();

//    delete[] expectedOutputs;
//    delete[] lastPerception;
//    delete[] perception;
    delete sgd;
    delete net;
    delete env;
    delete cl;

    return 0;
}
