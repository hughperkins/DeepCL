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

    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->insert();
    net->print();
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->epochMaker()
           ->learningRate(3)->batchSize(4)->numExamples(4)
           ->inputData(ldc.data)->expectedOutputs(ldc.expectedResults)
           ->run();
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
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
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
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
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
    ldc.applyAndGate();
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased()->insert();
    for( int epoch = 0; epoch < 10; epoch++ ) {
        net->epochMaker()->learningRate(1)->batchSize(4)->numExamples(4)->inputData(ldc.data)
           ->expectedOutputs(ldc.expectedResults)->run();
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
    delete net;
}

int main( int argc, char *argv[] ) {
    Timer timer;
   
//    testAnd();
    //testOr();
//    testXor();
    testAndConvolve();

//    BoardPng::writeBoardsToPng( "testneuralnetmnist-1.png", results, min(N, 100), boardSize );


    return 0;
}


