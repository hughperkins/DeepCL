//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "utils/Timer.h"
#include "NeuralNet.h"

void checkAccuracy( int numImages, int numPlanes, int const*labels, float const*results ) {
    int correct = 0;
    for( int n = 0; n < numImages; n++ ) {
        double maxValue = -100000;
        int bestIndex = -1;
        for( int plane = 0; plane < numPlanes; plane++ ) {
            if( results[ n * numPlanes + plane ] > maxValue ) {
                bestIndex = plane;
                maxValue = results[ n * numPlanes + plane ];
            }
        }
        cout << "expected: " << labels[n] << " got " << bestIndex << endl;
        if( bestIndex == labels[n] ) {
            correct++;
        }
    }
    cout << " accuracy: " << correct << "/" << numImages << endl;
}

int main( int argc, char *argv[] ) {
    Timer timer;

    float *data = new float[2];
    data[0] = 0.5;
    data[1] = -0.5;
    int *labels = new int[2];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedResults = new float[4];
    expectedResults[0] = 1;
    expectedResults[1] = -1;
    expectedResults[2] = -1;
    expectedResults[3] = 1;
    NeuralNet *net = new NeuralNet(1, 1 );
    net->addConvolutional( 2, 3 );
    for( int epoch = 0; epoch < 4; epoch++ ) {
        net->doEpoch( 1, 2, 2, data, expectedResults );
        cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
        net->print();
        float const*results = net->layers[1]->getResults();
        checkAccuracy( 2, 2, labels, results );
    }
    float const*results = net->getResults( net->getNumLayers() - 1 );


    delete net;

    return 0;
}


