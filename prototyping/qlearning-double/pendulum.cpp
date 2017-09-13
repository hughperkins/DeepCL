#include "pendulum.h"

static const char * dbus_path = "/gym/pendulum/env";
static const char * dbus_name = "gym.pendulum.env.service";

Pendulum::Pendulum(DBus::Connection& connection)
    : DBus::ObjectProxy(connection, dbus_path, dbus_name)
{
    cl = new EasyCL();
    net = new NeuralNet( cl );
    sgd = SGD::instance( cl, 0.01f, 0.0f );

    //input
    net->addLayer(InputLayerMaker::instance()->imageSize(1)->numPlanes(numPlanes));
    //h
    net->addLayer(FullyConnectedMaker::instance()->numPlanes(9)->imageSize(1));
    net->addLayer(ActivationMaker::instance()->relu());
    net->addLayer(FullyConnectedMaker::instance()->numPlanes(27)->imageSize(1));
    net->addLayer(ActivationMaker::instance()->sigmoid());
    net->addLayer(FullyConnectedMaker::instance()->numPlanes(18)->imageSize(1));
    net->addLayer(ActivationMaker::instance()->tanh());
    //output
    net->addLayer(FullyConnectedMaker::instance()->imageSize(1)->numPlanes(numActions));
    net->addLayer(SquareLossMaker::instance());
}

Pendulum::~Pendulum()
{
    delete sgd;
    delete net;
    delete cl;
}

void Pendulum::getPerception(float *perception)
{
    arrayZero(perception, numPlanes);
    auto itr = ob.begin();
    for (int i = 0; i < numPlanes; ++i) {
        if(i==2)
            perception[i] = (1.0f/8) * (*(itr++));
        else
            perception[i] = *(itr++);
    }
}

void Pendulum::reset()
{
    ob = gym::pendulum::env_proxy::reset();
}

float Pendulum::act(int index)
{
    double floatAction = (index + 0.5f - (numActions/2.0f)) * (4.0f/(numActions-1));
    double reward;

    ob.clear();
    step(floatAction, ob, reward, done);
    render();

    reward /= 10;

    std::cout << "action:" << floatAction << "," << index << " reward:" << reward << " done:" << done << std::endl;
    return reward;
}
