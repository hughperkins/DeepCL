#include <iostream>
#include <random>
//#include <thread>
//#include <chrono>
#include <vector>

#include "ScenarioImage.h"

#include "NeuralNet.h"

#include "array_helper.h"

#include "QLearner.h"

using namespace std;

int main( int argc, char *argv[] ) {
//    ScenarioImage scenario;

    ScenarioImage *scenario = new ScenarioImage( 7, true);

    NeuralNet *net = new NeuralNet();

    const int size = scenario->getPerceptionSize();
    const int planes = scenario->getPerceptionPlanes();
    const int numActions = scenario->getNumActions();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(planes)->imageSize(size) );
    net->addLayer( ConvolutionalMaker::instance()->filterSize(5)->numFilters(8)->biased()->padZeros()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->filterSize(5)->numFilters(8)->biased()->padZeros()->relu() );
    net->addLayer( FullyConnectedMaker::instance()->imageSize(1)->numPlanes(100)->biased()->tanh() );
    net->addLayer( FullyConnectedMaker::instance()->imageSize(1)->numPlanes(numActions)->linear()->biased() );
    net->addLayer( SquareLossMaker::instance() );
    net->print();

    scenario->setNet( net ); // used by the printQRepresentation method

    QLearner qLearner( scenario, net );
    qLearner.run();
    
//    delete[] expectedOutputs;
//    delete[] lastPerception;
//    delete[] perception;
    delete net;
    delete scenario;
    
    return 0;
}

