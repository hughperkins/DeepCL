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
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->biased()->insert();
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

void testAndRelu() {
    cout << "And" << endl;
    LogicalDataCreator ldc( new ReluActivation() );
    ldc.applyAndGate();

    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->relu()->biased()->insert();
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
    net->print();
    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
    assertEquals( numCorrect, ldc.N );
    delete net;
}

void testAndLinear() {
    cout << "And" << endl;
    LogicalDataCreator ldc( new ReluActivation() );
    ldc.applyAndGate();

    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->linear()->biased()->insert();
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
    net->print();
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
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->biased()->insert();
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
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->biased()->insert();
    net->fullyConnectedMaker()->planes(2)->boardSize(1)->biased()->insert();
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

//      n=0       n=1       n=2       n=3
//input plane0=-1 plane0=-1 plane0= 1 plane0=1
//      plane1=-1 plane1= 1 plane1=-1 plane1=1

//layer1 plane0=1 "planes both -1"
//     weights=plane0*(-1)+plane1*(-1)
//       plane1=0 "planes both 1"
//     weights=plane0*(1)+plane1*(1)

//layer2 plane0=0 "planes not both -1 and planes not both 1"
//      weights = plane0*(-1) + plane1*(-1)
//      plane1=1 "planes both -1 or planes both 1"
//      weights = plane0*(1) + plane1*(1)
void testXorConvolve() {
    cout << "Xor, convolve" << endl;
//    LogicalDataCreator ldc(new TanhActivation());
//    ldc.applyXorGate();

    int boardSize = 1;
    int inPlanes = 2;
    int numExamples = 4;
    int filterSize = 1;
    float data[] = { -1, -1,
                     -1, 1,
                     1, -1,
                     1, 1 };
    float layer1weights[] = {  // going to preset these, and at least show the network is stable, and gives the correct
         -1,-1,                      // result...
         1, 1,
    };
    float layer1bias[] = {
       0,
       0
    };
    float layer2weights[] = {
        1, 1,
        -1, -1
    };
    float layer2bias[] = {
       0,
       100
    };
    float expectedResults[] = {
        1, 0,
        0, 1,
        0, 1,
        1, 0
    };
    int labels[] = {
        0,
        1,
        1,
        0
    };

    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased(1)->relu()->insert();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased(1)->relu()->insert();
    cout << "hand-setting weights..." << endl;
    for( int i = 0; i < 4; i++ ) {
        net->layers[1]->weights[i] = layer1weights[i];
        net->layers[2]->weights[i] = layer2weights[i];
    }
    for( int i = 0; i < 2; i++ ) {
        dynamic_cast<ConvolutionalLayer *>(net->layers[1])->biasWeights[i] = layer1bias[i];
        dynamic_cast<ConvolutionalLayer *>(net->layers[2])->biasWeights[i] = layer2bias[i];
    }
    net->printWeights();
    net->setBatchSize(4);
    net->propagate( data );
//    net->print();
    for( int epoch = 0; epoch < 0; epoch++ ) {
        net->epochMaker()->learningRate(1)->batchSize(numExamples)->numExamples(numExamples)->inputData(data)
           ->expectedOutputs(expectedResults)->run();
        cout << "Loss L " << net->calcLoss(expectedResults) << endl;
    }
    net->printOutput();
    cout << "Loss L " << net->calcLoss(expectedResults) << endl;
    AccuracyHelper::printAccuracy( numExamples, 2, labels, net->getResults() );
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
    if( testNum == 8 ) testAndRelu();
    if( testNum == 9 ) testAndLinear();
    
//    BoardPng::writeBoardsToPng( "testneuralnetmnist-1.png", results, min(N, 100), boardSize );


    return 0;
}


