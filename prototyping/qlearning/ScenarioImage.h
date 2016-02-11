#include <iostream>
#include <string>
#include <algorithm>
#include <random>

#include "qlearning/Scenario.h"

class NeuralNet;

#define STATIC static
#define VIRTUAL virtual

// represents a small square world, with one apple,
// that might be always in the centre (if appleMoves is false)
// or in a random location (if appleMoves is true)
// - our hero is at location (posX,posY)
// - apple is at (appleX,appleY)
// - size of world is 'size' (world is square)
// two output planes are provided in the perception:
// - plane 0 shows where is our hero (0 everywhere, except 1 where hero is)
// - plane 1 shows where is the apple (0 everywhere, 1 where apple is)
// rewards given:
// - hit wall: -0.5
// - get apple: +1.0
// - make a move: -0.1
class ScenarioImage : public Scenario {
public:
    NeuralNet *net;
    const int size;
    bool appleMoves; // does apple move when reset?

    int posX;
    int posY;
    int appleX;
    int appleY;
    int game;
    int numMoves;

    int width;
    int height;
    bool finished;
    std::mt19937 myrand;

//    char**world;
    
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ScenarioImage( int size, bool appleMoves );
    void setNet( NeuralNet *net );
    void printQRepresentation();
    VIRTUAL void print();
    VIRTUAL ~ScenarioImage();
    VIRTUAL int getNumActions();
    VIRTUAL float act( int index );  // returns reward
    VIRTUAL bool hasFinished();
    VIRTUAL int getPerceptionSize();
    VIRTUAL int getPerceptionPlanes();
    VIRTUAL void getPerception( float *perception );
    VIRTUAL int getWorldSize();
    VIRTUAL void reset();

    // [[[end]]]
};

