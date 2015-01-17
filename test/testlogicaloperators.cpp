//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "gtest/gtest.h"

#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "LogicalDataCreator.h"
#include "test/myasserts.h"

//TEST( testlogicaloperators, DISABLED_FullyConnected_Biased_Tanh_And_1layer ) {
////    cout << "And" << endl;
//    LogicalDataCreator ldc;
//    ldc.applyAndGate();

//    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
//    net->fullyConnectedMaker()->planes(2)->boardSize(1)->biased()->tanh()->insert();
////    net->print();
//    for( int epoch = 0; epoch < 100; epoch++ ) {
//        net->epochMaker()
//           ->learningRate(1)->batchSize(4)->numExamples(4)
//           ->inputData(ldc.data)->expectedOutputs(ldc.expectedResults)
//           ->run();
//        if( epoch % 20 == 0 ) {
//            cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
//            AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        }
////        net->printWeights();
//    }
////    net->print();
//    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
//    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
//    assertEquals( numCorrect, ldc.N );

//    float loss = net->calcLoss(ldc.expectedResults);
//    cout << "loss, E, " << loss << endl;
//    assertLessThan( 0.4, loss );

//    delete net;
//}

//TEST( testlogicaloperators, DISABLED_FullyConnected_1layer_biased_tanh_Or ) {
//    cout << "Or" << endl;
//    LogicalDataCreator ldc;
////    ldc.applyAndGate();
//    ldc.applyOrGate();
////    NeuralNet *net = new NeuralNet(2, 1 );
//    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
//    net->fullyConnectedMaker()->planes(2)->boardSize(1)->biased()->tanh()->insert();
//    for( int epoch = 0; epoch < 10; epoch++ ) {
//        net->doEpoch( 5, 4, 4, ldc.data, ldc.expectedResults );
//        if( epoch % 5 == 0 ) {
//            cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
//            AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        }
////        net->printWeights();
//    }

//    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
//    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
//    assertEquals( numCorrect, ldc.N );

//    float loss = net->calcLoss(ldc.expectedResults);
//    cout << "loss, E, " << loss << endl;
//    assertLessThan( 0.4, loss );

//    delete net;
//}

//TEST( testlogicaloperators, DISABLED_FullyConnected_2layer_Xor ) {
////    cout << "Xor" << endl;
//    LogicalDataCreator ldc;
//    ldc.applyXorGate();
////    NeuralNet *net = new NeuralNet(2, 1 );
//    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
//    net->fullyConnectedMaker()->planes(2)->boardSize(1)->biased()->insert();
//    net->fullyConnectedMaker()->planes(2)->boardSize(1)->biased()->insert();
//    float weights1[] = {-1.04243, 0.251409, -0.806014, -0.0268563};
//    float weights2[] = {0.107038, -0.144079, 0.1492, -0.395718};
//    float biasWeights1[] = {-0.415169, 0.536681};
//    float biasWeights2[] = {-0.0480136, 0.167825};
//    net->initWeights( 1, weights1, biasWeights1 );
//    net->initWeights( 2, weights2, biasWeights2 );
//    for( int epoch = 0; epoch < 200; epoch++ ) {
//        net->doEpoch( 1, 4, 4, ldc.data, ldc.expectedResults );
////        net->printWeightsAsCode();
////        net->printBiasWeightsAsCode();

//        if( epoch % 50 == 0 ) {
//            float loss = net->calcLoss(ldc.expectedResults);
//            cout << "loss, E, " << loss << endl;
//        }
////        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
////        net->printWeights();
////        if( loss < 0.00001 ) {
////            break;
////        }
//    }
//    cout << " Loss L " << net->calcLoss(ldc.expectedResults) << endl;
//    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
//    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
////    if( numCorrect != ldc.N ) {
////        net->print();
////    }
//    assertEquals( numCorrect, ldc.N );

//    float loss = net->calcLoss(ldc.expectedResults);
//    cout << "loss, E, " << loss << endl;
//    assertLessThan( 0.00001, loss );

//    delete net;
//}

TEST( testlogicaloperators, DISABLED_Convolve_1layer_And_Nobias ) {
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

TEST( testlogicaloperators, Convolve_1layer_biased_And ) {
    cout << "And" << endl;
    LogicalDataCreator ldc;
    ldc.applyAndGate();
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased(1)->insert();
    net->squareLossMaker()->insert();
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()->learningRate(1)->batchSize(4)->numExamples(4)->inputData(ldc.data)
           ->expectedOutputs(ldc.expectedResults)->run();
        if( epoch % 5 == 0 ) cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
//        net->printWeights();
    }
//        net->print();
    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
    assertEquals( numCorrect, ldc.N );

    float loss = net->calcLoss(ldc.expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.4, loss );

    delete net;
}

TEST( testlogicaloperators, Convolve_1layerbiased_Or ) {
    cout << "Or, convolve" << endl;
    LogicalDataCreator ldc;
    ldc.applyOrGate();
    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(1)->biased(1)->insert();
    net->squareLossMaker()->insert();
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()->learningRate(1)->batchSize(4)->numExamples(4)->inputData(ldc.data)
           ->expectedOutputs(ldc.expectedResults)->run();
        if( epoch % 5 == 0 ) cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
//        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
//        net->printWeights();
    }
//        net->print();
        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );

    float loss = net->calcLoss(ldc.expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.4, loss );

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
TEST( testlogicaloperators, Convolve_2layers_relu_Xor ) {
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
    float layer1weights[] = {  // going to preset these, to near an optimal solution,
                              //  and at least show the network is stable, and gives the correct
         -0.4,-0.55,                      // result...
         0.52, 0.53,
    };
    float layer1bias[] = {
       0.1,
       -0.1
    };
    float layer2weights[] = {
        1.1, 0.9,
        -0.8, -1.2
    };
    float layer2bias[] = {
       0.1,
       1.1
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
    net->squareLossMaker()->insert();
    cout << "hand-setting weights..." << endl;
    net->initWeights( 1, layer1weights, layer1bias );
    net->initWeights( 2, layer2weights, layer2bias );
//    net->printWeights();
//    net->setBatchSize(4);
//    net->propagate( data );
//    net->print();
    for( int epoch = 0; epoch < 200; epoch++ ) {
        net->epochMaker()->learningRate(0.1f)->batchSize(numExamples)->numExamples(numExamples)->inputData(data)
           ->expectedOutputs(expectedResults)->run();
        if( epoch % 5 == 0 ) cout << "Loss L " << net->calcLoss(expectedResults) << endl;
    }
    net->print();
    AccuracyHelper::printAccuracy( numExamples, 2, labels, net->getResults() );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.0000001, loss );

    delete net;
}

//TEST( testlogicaloperators, DISABLED_Convolve_1layer_relu_biased_And ) {
//    cout << "And" << endl;
//    LogicalDataCreator ldc( new ReluActivation() );
//    ldc.applyAndGate();

//    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
//    net->fullyConnectedMaker()->planes(2)->boardSize(1)->relu()->biased()->insert();
//    net->print();
//    for( int epoch = 0; epoch < 10; epoch++ ) {
//        net->epochMaker()
//           ->learningRate(3)->batchSize(4)->numExamples(4)
//           ->inputData(ldc.data)->expectedOutputs(ldc.expectedResults)
//           ->run();
//        cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
//        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
////        net->printWeights();
//    }
//    net->print();
//    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
//    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
//    assertEquals( numCorrect, ldc.N );
//    delete net;

//}

//TEST( testlogicaloperators, DISABLED_FullyConnected_1layer_biased_linear_And ) {
//    cout << "And" << endl;
//    LogicalDataCreator ldc( new ReluActivation() );
//    ldc.applyAndGate();

//    NeuralNet *net = NeuralNet::maker()->planes(2)->boardSize(1)->instance();
//    net->fullyConnectedMaker()->planes(2)->boardSize(1)->linear()->biased()->insert();
////    net->print();
//    for( int epoch = 0; epoch < 20; epoch++ ) {
//        net->epochMaker()
//           ->learningRate(3)->batchSize(4)->numExamples(4)
//           ->inputData(ldc.data)->expectedOutputs(ldc.expectedResults)
//           ->run();
//                
//        if( epoch % 5 == 0 ) cout << "Loss L " << net->calcLoss(ldc.expectedResults) << endl;
////        AccuracyHelper::printAccuracy( ldc.N, 2, ldc.labels, net->getResults() );
////        net->printWeights();
//    }
////    net->print();
//    int numCorrect = AccuracyHelper::calcNumRight( ldc.N, 2, ldc.labels, net->getResults() );
//    cout << "accuracy: " << numCorrect << "/" << ldc.N << endl;
//    assertEquals( numCorrect, ldc.N );

//    float loss = net->calcLoss(ldc.expectedResults);
//    cout << "loss, E, " << loss << endl;
//    assertLessThan( 0.4, loss );

//    delete net;
//}


