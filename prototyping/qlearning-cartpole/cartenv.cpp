#include "cartenv.h"

#include "qlearning/array_helper.h"

int CartEnv::getPerceptionSize()
{
    return 1;
}

int CartEnv::getPerceptionPlanes()
{
    return 4;
}

void CartEnv::getPerception(float *perception)
{
    arrayZero(perception, perceptionSize);
    int i=0;
    for (auto itr = ob.begin();  itr != ob.end(); itr++) {
        perception[i++] = *itr;
    }
}

void CartEnv::reset()
{
    std::cout <<"game:" << game << " step:" << stepCount;
    if(game)
        std::cout << " avg:" << this->reward / game << std::endl;

    stepCount = 0;
    if(stepBestKeep >= 5)
    {
        stepBest = this->reward / game;
        stepBestKeep = 0;
    }
    game++;
    gym::cart::env_proxy::reset();
}

int CartEnv::getNumActions()
{
    return 2;
}

float CartEnv::act(int index)
{
    double reward;
    ob.clear();
    this->step(index, ob, reward, done);
    this->render();

    stepCount++;
    this->reward = this->reward + reward;

    if(stepBest < stepCount)
    {
        stepBest = stepCount;
        stepBestKeep = 0;
    }
    else
        stepBestKeep++;

    if(done)
        return -1;
    else if(stepBest < stepCount)
        return 1;
    else
        return 0.05;

}

bool CartEnv::hasFinished()
{
    return done;
}

void CartEnv::test()
{

}
