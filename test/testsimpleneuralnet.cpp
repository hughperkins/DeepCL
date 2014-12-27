//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "utils/Timer.h"
#include "NeuralNet.h"

int main( int argc, char *argv[] ) {
    Timer timer;

    float *data = new float[2];
//    float *labels = new float[2];
    data[0] = 0.5;
    data[1] = -0.5;
//    labels[0] = 0;
//    labels[1] = 1;
    float *expectedResults = new float[4];
    expectedResults[0] = 1;
    expectedResults[1] = -1;
    expectedResults[2] = -1;
    expectedResults[3] = 1;
    NeuralNet *net = new NeuralNet(1, 1 );
    net->addFullyConnected( 2 );
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->doEpoch( 0.1, 2, 2, data, expectedResults );
        //net->print();
    }
    float const*results = net->getResults( net->getNumLayers() - 1 );


//    BoardPng::writeBoardsToPng( "testneuralnetmnist-1.png", results, min(N, 100), boardSize );

    delete net;

    return 0;
}


