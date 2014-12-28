//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "LogicalDataCreator.h"
#include "test/asserts.h"

void testAnd() {
    cout << "And" << endl;
    LogicalDataCreator ldc;
    ldc.applyAndGate();

    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->insert();
    net->print();
    for( int epoch = 0; epoch < 10; epoch++ ) {
        net->epochMaker()
           ->learningRate(3)->batchSize(4)->numExamples(4)
           ->inputData(ldc.data)->expectedOutputs(ldc.expectedResults)
           ->run();
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
    assertEquals( numCorrect, ldc.N );
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
    for( int epoch = 0; epoch < 100000; epoch++ ) {
        net->doEpoch( 0.1, 4, 4, ldc.data, ldc.expectedResults );
//        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
    cout << " Loss L " << net->calcLoss(ldc.expectedResults) << endl;
    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
    if( numCorrect != ldc.N ) {
        net->print();
    }
    assertEquals( numCorrect, ldc.N );
    delete net;
}

void testAndConvolveNoBias() {
    cout << "And" << endl;
    LogicalDataCreator ldc;
    ldc.applyAndGate();
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased(0)->insert();
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()->learningRate(4)->batchSize(4)->numExamples(4)->inputData(ldc.data)
           ->expectedOutputs(ldc.expectedResults)->run();
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
//        net->printWeights();
    }
//    net->print();
    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
    assertEquals( numCorrect, ldc.N );
    delete net;
}

void testAndConvolveBiased() {
    cout << "And" << endl;
    LogicalDataCreator ldc;
    ldc.applyAndGate();
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased(1)->insert();
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()->learningRate(4)->batchSize(4)->numExamples(4)->inputData(ldc.data)
           ->expectedOutputs(ldc.expectedResults)->run();
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
//        net->printWeights();
    }
        net->print();
    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
    assertEquals( numCorrect, ldc.N );
    delete net;
}

void testOrConvolve() {
    cout << "Or, convolve" << endl;
    LogicalDataCreator ldc;
    ldc.applyOrGate();
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased(1)->insert();
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->epochMaker()->learningRate(1)->batchSize(4)->numExamples(4)->inputData(ldc.data)
           ->expectedOutputs(ldc.expectedResults)->run();
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
        net->print();
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
    delete net;
}

void testXorConvolve() {
    cout << "Xor, convolve" << endl;
    LogicalDataCreator ldc;
    ldc.applyAndGate();
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased()->insert();
    for( int epoch = 0; epoch < 10; epoch++ ) {
        net->epochMaker()->learningRate(1)->batchSize(4)->numExamples(4)->inputData(ldc.data)
           ->expectedOutputs(ldc.expectedResults)->run();
        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
        net->printWeights();
    }
    delete net;
}

int main( int argc, char *argv[] ) {
    Timer timer;

   int testNum = -1;
   if( argc == 2 ) testNum = atoi( argv[1] );

    if( testNum == -1 ) {
        for( int i = 0; i < 10; i++ ) {
            testAnd();
//            testXor();
            testAndConvolveBiased();
        //    testOrConvolve();
        }
    }   

    if( testNum == 1 ) testAnd();
    if( testNum == 2 ) testOr();
    if( testNum == 3 ) testXor();
    if( testNum == 4 ) testAndConvolveNoBias();
    if( testNum == 5 ) testAndConvolveBiased();
    if( testNum == 6 ) testOrConvolve();
    if( testNum == 7 ) testXorConvolve();
    
//    BoardPng::writeBoardsToPng( "testneuralnetmnist-1.png", results, min(N, 100), boardSize );


    return 0;
}


