#include <vector>
#include <stdexcept>

#include "ScenarioImage.h"

#include "util/stringhelper.h"
#include "qlearning/array_helper.h"

#include "net/NeuralNet.h"

using namespace std;

#undef STATIC
#define STATIC
#undef VIRTUAL
#define VIRTUAL

ScenarioImage::ScenarioImage(int size, bool appleMoves) :
        net(0), // this is simply used by the showQRepresentation method
        size(size),
        appleMoves(appleMoves) {
//    size = 7;
    posX = 1;
    posY = 1;
    appleX = 2;
    appleY = 2;
    game = 0;
    numMoves = 0;
//    appleMoves = true; // does apple move when reset?

    reset();
    print();
}
void ScenarioImage::setNet(NeuralNet *net) {
    this->net = net;
}
void ScenarioImage::printQRepresentation() {
    ScenarioImage *scenario = this;
    cout << "q directions:" << endl;
    int size = scenario->getPerceptionSize();
    float *input = new float[ size * size * 2 ];
    arrayZero(input, size * size * 2);
    input[ scenario->appleY * size + scenario->appleX ] = 1;
    for(int y = 0; y < size; y++) {
        string thisLine = "";
        for(int x = 0; x < size; x++) {
            float highestQ = 0;
            int bestAction = 0;
            input[ size * size + y * size + x ] = 1;
            net->forward(input);
            input[ size * size + y * size + x ] = 0;
            float const*output = net->getOutput();
            for(int action = 0; action < 4; action++) {
                float thisQ = output[action];
                if(action == 0 || thisQ > highestQ) {
                    highestQ = thisQ;
                    bestAction = action;
                }
            }
            if(bestAction == 0) {
                thisLine += ">";
            } else if(bestAction == 1) {
                thisLine += "<";
            } else if(bestAction == 2) {
                thisLine += "V";
            } else {
                thisLine += "^";
            }
        }
        cout << thisLine << endl;
    }
    delete[]input;
}

VIRTUAL void ScenarioImage::print() {
    for(int y = 0; y < size; y++) {
        string line = "";
        for(int x = 0; x < size; x++) {
            if(x == posX && y == posY) {
                line += "X";
            } else if(x == appleX && y == appleY) {
                line += "O";
            } else {
                line += ".";
            }
        }
        cout << line << endl;
    }
}
VIRTUAL ScenarioImage::~ScenarioImage() {
}
VIRTUAL int ScenarioImage::getNumActions() {
    return 4;
}
VIRTUAL float ScenarioImage::act(int index) { // returns reward
    numMoves++;
    int dx = 0;
    int dy = 0;
    switch(index) {
        case 0:
            dx = 1;
            break;
        case 1:
            dx = -1;
            break;
        case 2:
            dy = 1;
            break;
        case 3:
            dy = -1;
            break;
    }
    int newX = posX + dx;
    int newY = posY + dy;
    if(newX < 0 || newX >= size || newY < 0 || newY >= size) {
        return -0.5f;
    }
    if(newX == appleX && newY == appleY) {
        finished = true;
        posX = newX;
        posY = newY;
        return 1;
    } else {
        posX = newX;
        posY = newY;
        return -0.1f;
    }
}
VIRTUAL bool ScenarioImage::hasFinished() {
    return finished;
}
VIRTUAL int ScenarioImage::getPerceptionSize() {
    return size;
}
VIRTUAL int ScenarioImage::getPerceptionPlanes() {
    return 2;
}
VIRTUAL void ScenarioImage::getPerception(float *perception) {
    for(int i = 0; i < size * size * 2; i++) {
        perception[i] = 0;
    }
    perception[appleY * size + appleX] = 1;
    perception[size * size + posY * size + posX] = 1; 
}
VIRTUAL int ScenarioImage::getWorldSize() {
    return size;
}
VIRTUAL void ScenarioImage::reset() {
    if(net != 0) {
        this->print();
        this->printQRepresentation();
        cout << "game: " << game << " moves: " << numMoves << endl;
    }
    if(appleMoves) {
        appleX = myrand() % size;
        appleY = myrand() % size;
    } else {
        appleX = appleY = size / 2;
    }
    finished = false;
    bool sampledOnce = false;
    while(!sampledOnce || (posX == appleX && posY == appleY)) {
        posX = myrand() % size;
        posY = myrand() % size;
        sampledOnce = true;
    }
    game++;
    numMoves = 0;
}

