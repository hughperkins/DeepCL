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
        perception[i] = (*itr);
        i++;
    }
}

void CartEnv::reset()
{
    std::cout <<"game:" << game << " step:" << stepCount;
    if(game)
        avg = this->reward / game;
    std::cout << " avg:" << avg << std::endl;

    stepCount = 0;
    if(stepBestKeep >= 5)
    {
        stepBest = avg;
        stepBestKeep = 0;
    }
    game++;
    ob = gym::cart::env_proxy::reset();
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
    {
        if(stepCount < avg)
            return -1;
        else
            return -0.85;
    }
    else if(stepBest < stepCount)
        return 1 * reward;
    else if(avg < stepCount)
        return 0.07 * reward;
    else
        return 0.01 * reward;

}

bool CartEnv::hasFinished()
{
    return done;
}

void CartEnv::test()
{

}
