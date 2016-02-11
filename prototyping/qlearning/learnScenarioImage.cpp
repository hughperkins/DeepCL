#include <iostream>
#include <random>
//#include <thread>
//#include <chrono>
#include <vector>

#include "ScenarioImage.h"
#include "DeepCL.h"
#include "qlearning/array_helper.h"
#include "qlearning/QLearner.h"

using namespace std;

int main( int argc, char *argv[] ) {
//    ScenarioImage scenario;

    ScenarioImage *scenario = new ScenarioImage( 5, true);

    EasyCL *cl = new EasyCL();
    NeuralNet *net = new NeuralNet( cl );
    SGD *sgd = SGD::instance( cl, 0.1f, 0.0f );

    const int size = scenario->getPerceptionSize();
    const int planes = scenario->getPerceptionPlanes();
    const int numActions = scenario->getNumActions();
    net->addLayer( InputLayerMaker::instance()->numPlanes(planes)->imageSize(size) );
    net->addLayer( ConvolutionalMaker::instance()->filterSize(5)->numFilters(8)->biased()->padZeros() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->filterSize(5)->numFilters(8)->biased()->padZeros() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( FullyConnectedMaker::instance()->imageSize(1)->numPlanes(100)->biased() );
        net->addLayer( ActivationMaker::instance()->tanh() );
    net->addLayer( FullyConnectedMaker::instance()->imageSize(1)->numPlanes(numActions)->biased() );
    net->addLayer( SquareLossMaker::instance() );
    net->print();

    scenario->setNet( net ); // used by the printQRepresentation method

    QLearner qLearner( sgd, scenario, net );
    qLearner.run();
    
//    delete[] expectedOutputs;
//    delete[] lastPerception;
//    delete[] perception;
    delete sgd;
    delete net;
    delete scenario;
    delete cl;
    
    return 0;
}

