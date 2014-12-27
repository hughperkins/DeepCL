//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "LogicalDataCreator.h"

void testAnd() {
    cout << "And" << endl;
    LogicalDataCreator ldc;
    ldc.applyAndGate();

    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->new();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->insert();
    net->print();
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->doEpoch( 3, 4, 4, ldc.data, ldc.expectedResults );
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
    delete net;
}

void testOr() {
    cout << "Or" << endl;
    LogicalDataCreator ldc;
//    ldc.applyAndGate();
    ldc.applyOrGate();
//    NeuralNet *net = new NeuralNet(2, 1 );
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->new();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->insert();
    for( int epoch = 0; epoch < 10; epoch++ ) {
        net->doEpoch( 5, 4, 4, ldc.data, ldc.expectedResults );
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
    delete net;
}

void testXor() {
    cout << "Xor" << endl;
    LogicalDataCreator ldc;
    ldc.applyXorGate();
//    NeuralNet *net = new NeuralNet(2, 1 );
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->new();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->insert();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->insert();
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->doEpoch( 1, 4, 4, ldc.data, ldc.expectedResults );
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
    delete net;
}


void testAndConvolve() {
    cout << "And" << endl;
    LogicalDataCreator ldc;
//    ldc.applyAndGate();
    ldc.applyAndGate();
//    NeuralNet *net = new NeuralNet(2, 1 );
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->new();
    net->convolutionalMaker()->filters(2)->filterSize(1)->insert();
    for( int epoch = 0; epoch < 5; epoch++ ) {
        net->doEpoch( 1, 4, 4, ldc.data, ldc.expectedResults );
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
    delete net;
}

int main( int argc, char *argv[] ) {
    Timer timer;
   
    testAnd();
    //testOr();
//    testXor();
  //  testAndConvolve();

//    BoardPng::writeBoardsToPng( "testneuralnetmnist-1.png", results, min(N, 100), boardSize );


    return 0;
}


