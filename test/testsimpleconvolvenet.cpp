//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>

#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "test/asserts.h"

using namespace std;

void test1() {
    Timer timer;
    float *data = new float[2];
    data[0] = 0.5;
    data[1] = -0.5;
    int *labels = new int[2];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedResults = new float[4];
    expectedResults[0] = 0.5;
    expectedResults[1] = -0.5;
    expectedResults[2] = -0.5;
    expectedResults[3] = 0.5;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased()->insert();
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->epochMaker()
            ->learningRate(1)
            ->batchSize(2)
            ->numExamples(2)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
//        cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
//        net->print();
        float const*results = net->getResults();
//        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( 2, 2, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    assertEquals( numCorrect, 2 );

    delete net;
}

void test2() {
    Timer timer;
    float data[] = { 0.5, 0.5, 0.5,
                    -0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5,
    
                   0.5, 0.5, 0.5,
                   0.5, -0.5, 0.5,
                   0.5, 0.5, 0.5,

                    -0.5, -0.5, -0.5,
                    -0.5, 0.5, -0.5,
                    -0.5, -0.5, -0.5,
    
                   -0.5, -0.5, -0.5,
                   0.5, -0.5, -0.5,
                   -0.5, -0.5, -0.5
 };

    int *labels = new int[4];
    labels[0] = 0;
    labels[1] = 1;
    labels[2] = 0;
    labels[3] = 1;
    float *expectedResults = new float[8];
    expectedResults[0] = 0.5;
    expectedResults[1] = -0.5;
    expectedResults[2] = -0.5;
    expectedResults[3] = 0.5;
    expectedResults[4] = 0.5;
    expectedResults[5] = -0.5;
    expectedResults[6] = -0.5;
    expectedResults[7] = 0.5;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(3)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(3)->biased()->insert();
    float const*results = 0;
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()
            ->learningRate(1)
            ->batchSize(4)
            ->numExamples(4)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        results = net->getResults();
        AccuracyHelper::printAccuracy( 4, 2, labels, results );
    }
    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, results );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    assertEquals( numCorrect, 4 );

    delete net;
}

void test3_relu() {
    Timer timer;
    float *data = new float[2];
    data[0] = 0.5;
    data[1] = -0.5;
    int *labels = new int[2];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedResults = new float[4];
    expectedResults[0] = 1;
    expectedResults[1] = 0;
    expectedResults[2] = 0;
    expectedResults[3] = 1;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased()->relu()->insert();
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->epochMaker()
            ->learningRate(1)
            ->batchSize(2)
            ->numExamples(2)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
//        cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
//        net->print();
        float const*results = net->getResults();
//        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }
    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( 2, 2, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    assertEquals( numCorrect, 2 );

    delete net;
}

void test4_relu() {
    Timer timer;
    float data[] = { 0.5, 0.5, 0.5,
                    -0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5,
    
                   0.5, 0.5, 0.5,
                   0.5, -0.5, 0.5,
                   0.5, 0.5, 0.5,

                    -0.5, -0.5, -0.5,
                    -0.5, 0.5, -0.5,
                    -0.5, -0.5, -0.5,
    
                   -0.5, -0.5, -0.5,
                   0.5, -0.5, -0.5,
                   -0.5, -0.5, -0.5
 };

    int *labels = new int[4];
    labels[0] = 0;
    labels[1] = 1;
    labels[2] = 0;
    labels[3] = 1;
    float *expectedResults = new float[8];
    expectedResults[0] = 1;
    expectedResults[1] = 0;
    expectedResults[2] = 0;
    expectedResults[3] = 1;
    expectedResults[4] = 1;
    expectedResults[5] = 0;
    expectedResults[6] = 0;
    expectedResults[7] = 1;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(3)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(3)->biased()->relu()->insert();
    float const*results = 0;
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()
            ->learningRate(1)
            ->batchSize(4)
            ->numExamples(4)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        results = net->getResults();
        AccuracyHelper::printAccuracy( 4, 2, labels, results );
    }
    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, results );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    assertEquals( numCorrect, 4 );

    delete net;
}

void test5_linear() {
    Timer timer;
    float data[] = { 0.5, 0.5, 0.5,
                    -0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5,
    
                   0.5, 0.5, 0.5,
                   0.5, -0.5, 0.5,
                   0.5, 0.5, 0.5,

                    -0.5, -0.5, -0.5,
                    -0.5, 0.5, -0.5,
                    -0.5, -0.5, -0.5,
    
                   -0.5, -0.5, -0.5,
                   0.5, -0.5, -0.5,
                   -0.5, -0.5, -0.5
 };

    int *labels = new int[4];
    labels[0] = 0;
    labels[1] = 1;
    labels[2] = 0;
    labels[3] = 1;
    float *expectedResults = new float[8];
    expectedResults[0] = 1;
    expectedResults[1] = 0;
    expectedResults[2] = 0;
    expectedResults[3] = 1;
    expectedResults[4] = 1;
    expectedResults[5] = 0;
    expectedResults[6] = 0;
    expectedResults[7] = 1;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(3)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(3)->biased()->linear()->insert();
    float const*results = 0;
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()
            ->learningRate(1)
            ->batchSize(4)
            ->numExamples(4)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        results = net->getResults();
        AccuracyHelper::printAccuracy( 4, 2, labels, results );
    }
    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, results );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    assertEquals( numCorrect, 4 );

    delete net;
}

int main( int argc, char *argv[] ) {
    int testNum = -1;
    if( argc == 2 ) {
        testNum = atoi( argv[1] );
    }

    if( testNum == -1 ) {
        for( int i = 0; i < 10; i++ ) {
            test1();
            test2();
//            test3_relu();
//            test4_relu();
        }
    }

    if( testNum == -1 || testNum == 1 ) test1();
    if( testNum == -1 || testNum == 2 ) test2();
    if( testNum == -1 || testNum == 3 ) test3_relu();
    if( testNum == -1 || testNum == 4 ) test4_relu();
    if( testNum == -1 || testNum == 5 ) test5_linear();

    return 0;
}


