#ifndef CARTENV_H
#define CARTENV_H

#include "dbus-c++/dbus.h"
#include "CartEnvProxy.h"
#include "qlearning/Scenario.h"

static const char * dbus_path = "/gym/cart/env";
static const char * dbus_name = "gym.cart.env.service";

class CartEnv : gym::cart::env_proxy,
        public DBus::IntrospectableProxy,
        public DBus::ObjectProxy,
        public Scenario
{
    std::vector<double> ob;
    int game = 0;
    int stepCount;
    int stepBest = 0;
    int stepBestKeep;
    int reward = 0;
    int avg = 0;
    bool done = false;
    const int perceptionSize = getPerceptionSize() * getPerceptionSize() * getPerceptionPlanes();

public:
    CartEnv(DBus::Connection& connection)
        : DBus::ObjectProxy(connection, dbus_path, dbus_name) {
    }

    virtual int getPerceptionSize();
    virtual int getPerceptionPlanes();
    virtual void getPerception(float *perception);
    virtual void reset();
    virtual int getNumActions();
    virtual float act(int index);  // returns reward
    virtual bool hasFinished();

    void test();
};

#endif // CARTENV_H
