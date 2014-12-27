//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>

#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"

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
    expectedResults[0] = 1;
    expectedResults[1] = -1;
    expectedResults[2] = -1;
    expectedResults[3] = 1;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->insert();
    for( int epoch = 0; epoch < 50; epoch++ ) {
        net->epochMaker()
            ->learningRate(1)
            ->batchSize(2)
            ->numExamples(2)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
        net->print();
        float const*results = net->getResults();
        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }

    delete net;
}

void test2() {
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
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->insert();
    for( int epoch = 0; epoch < 2; epoch++ ) {
        net->epochMaker()
            ->learningRate(1)
            ->batchSize(2)
            ->numExamples(2)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
        net->print();
        float const*results = net->getResults();
        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }

    delete net;
}

int main( int argc, char *argv[] ) {

    test1();
    //test2();

    return 0;
}


