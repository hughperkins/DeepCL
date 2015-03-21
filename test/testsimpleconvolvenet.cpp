//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>

#include "gtest/gtest.h"

#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "test/myasserts.h"
#include "StatefulTimer.h"
#include "BatchLearner.h"
#include "NetLearner.h"

using namespace std;

TEST( testsimpleconvolvenet, imagesize1_planes2_filters2_unbiased_tanh ) {
    Timer timer;
    const float learningRate = 0.1f;
    const int batchSize = 2;
    float *data = new float[batchSize];
    data[0] = 0.5;
    data[1] = -0.5;
    int *labels = new int[batchSize];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedResults = new float[4];
    expectedResults[0] = 0.5;
    expectedResults[1] = -0.5;
    expectedResults[2] = -0.5;
    expectedResults[3] = 0.5;
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(1) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(0)->tanh() );
    net->addLayer( SquareLossMaker::instance() );;
    float weights1[] = {0.382147, -1.77522};
    net->initWeights(1, weights1);

    BatchLearner<float> batchLearner( net );
    for( int epoch = 0; epoch < 50; epoch++ ) {
        batchLearner.runEpochFromExpected( learningRate, batchSize, batchSize, data, expectedResults );
        if( epoch % 10 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
            float const*results = net->getResults();
            AccuracyHelper::printAccuracy( 2, 2, labels, results );
        }
    }

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( 2, 2, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    assertEquals( numCorrect, 2 );
    assertLessThan( 0.03, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize1_planes2_filters2_tanh ) {
    Timer timer;
    const float learningRate = 1.0f;
    const int batchSize = 2;
    float *data = new float[batchSize];
    data[0] = 0.5;
    data[1] = -0.5;
    int *labels = new int[batchSize];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedResults = new float[4];
    expectedResults[0] = 0.5;
    expectedResults[1] = -0.5;
    expectedResults[2] = -0.5;
    expectedResults[3] = 0.5;
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(1) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased()->tanh() );
    net->addLayer( SquareLossMaker::instance() );;
    float weights1[] = {0.382147, -1.77522};
    float biasweights1[] = {-1.00181, 0.891056};
    net->initWeights(1, weights1);
    net->initBiasWeights(1, biasweights1);

    BatchLearner<float> batchLearner( net );
    for( int epoch = 0; epoch < 30; epoch++ ) {
        batchLearner.runEpochFromExpected( learningRate, batchSize, batchSize, data, expectedResults );
        if( epoch % 10 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
            float const*results = net->getResults();
            AccuracyHelper::printAccuracy( 2, 2, labels, results );
        }
    }

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( 2, 2, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    assertEquals( numCorrect, 2 );
    assertLessThan( 0.01, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize3_n4_filtersize3_tanh ) {
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
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(3) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased() );
    net->addLayer( SquareLossMaker::instance() );;
    float weights1[] = {-0.171115, 0.28369, 0.201354, -0.496124, 0.391512, 0.120458, 0.396952, -0.1356, -0.319595, 0.251043, 0.318859, 0.220892, -0.480651, -0.51708, 0.2173, 0.365935, 0.304687, -0.712624};
    float biasWeights1[] = {0.375101, 0.00130748};
    net->initWeights(1, weights1);
    net->initBiasWeights(1, biasWeights1 );
    float const*results = 0;
    BatchLearner<float> batchLearner( net );
    for( int epoch = 0; epoch < 15; epoch++ ) {
        batchLearner.runEpochFromExpected( 0.4f, 4, 4, data, expectedResults );
//        net->printWeightsAsCode();
//        net->printBiasWeightsAsCode();
        if( epoch % 5 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
            results = net->getResults();
            AccuracyHelper::printAccuracy( 4, 2, labels, results );
        }
    }
//    net->print();
    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, results );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    assertEquals( numCorrect, 4 );
    assertLessThan( 0.0001, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize1_2planes_filtersize1_relu ) {
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
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(1) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased()->relu() );
    net->addLayer( SquareLossMaker::instance() );;
    float weights1[] = {-0.380177, -1.5738};
    float biasWeights1[] = {0.5, 0.0606055};
    net->initWeights( 1, weights1, biasWeights1 );
    BatchLearner<float> batchLearner( net );
    for( int epoch = 0; epoch < 5; epoch++ ) {
        batchLearner.runEpochFromExpected( 1.2f, 2, 2, data, expectedResults );
        if( epoch % 5 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
//            net->print();
    //        net->printWeightsAsCode();
    //        net->printBiasWeightsAsCode();
            float const*results = net->getResults();
            AccuracyHelper::printAccuracy( 2, 2, labels, results );
        }
    }
//    net->print();
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( 2, 2, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    assertEquals( numCorrect, 2 );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.001, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize3_n4_filtersize3_relu ) {
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
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(3) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(3)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased()->relu() );
    net->addLayer( SquareLossMaker::instance() );;
    float const*results = 0;
    float weights1[] = {0.0113327, 0.280063, -0.0584702, -0.503431, -0.37286, -0.457257, 0.29226, -0.360089, -0.273977, 0.530185, -0.460167, 0.489126, 0.141883, 0.179525, -0.18084, 0.412117, 0.0866731, -0.247958};
    float biasWeights1[] = {0.0418723, 0.158733};
    net->initWeights( 1, weights1, biasWeights1 );
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.4f)
            ->batchSize(4)
            ->numExamples(4)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        if( epoch % 5 == 0 ) {
            results = net->getResults();
    //        net->printWeightsAsCode();
    //        net->printBiasWeightsAsCode();
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
            AccuracyHelper::printAccuracy( 4, 2, labels, results );
        }
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, results );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    assertEquals( numCorrect, 4 );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.000001, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize3_n4_filtersize3_linear ) {
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
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(3) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(3)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased()->linear() );
    net->addLayer( SquareLossMaker::instance() );;
    float const*results = 0;
    float weights1[] = {0.715867, -0.428623, -0.281465, -0.736675, -0.224507, 0.335028, -0.384762, -0.213304, 0.679177, -0.170055, 0.335075, -0.572057, -0.175718, -0.410962, -0.175277, 0.536131, -0.0568329, -0.00297278};
    float biasWeights1[] = {0.5, 0.5};
    net->initWeights( 1, weights1, biasWeights1 );
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.09f)
            ->batchSize(4)
            ->numExamples(4)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
//        net->printWeightsAsCode();
//        net->printBiasWeightsAsCode();
        if( epoch % 5 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
            results = net->getResults();
            AccuracyHelper::printAccuracy( 4, 2, labels, results );
        }
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, results );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    assertEquals( numCorrect, 4 );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.001, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize1_n2_2layers_unbiased ) {
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
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(1) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(0) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(0) );
    net->addLayer( SquareLossMaker::instance() );;
    for( int epoch = 0; epoch < 30; epoch++ ) {
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

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.0001, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize1_n2_2layers_biased ) {
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
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(1) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
    net->addLayer( SquareLossMaker::instance() );;
float weights1[] = {1.12739, 1.21476};
float weights2[] = {-0.352846, 0.534554, -1.13343, -0.191175};
float biasWeights1[] = {0.971267, 1.42629};
float biasWeights2[] = {-0.071288, 0.443919};
    net->initWeights(1, weights1, biasWeights1 );
    net->initWeights(2, weights2, biasWeights2 );
    for( int epoch = 0; epoch < 30; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.4f)
            ->batchSize(2)
            ->numExamples(2)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        if( epoch % 5 == 0 ) {
//           net->printWeightsAsCode();
//            net->printBiasWeightsAsCode();
        cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
        }
//        net->print();
//        float const*results = net->getResults();
//        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }
//    net->print();

    StatefulTimer::dump(true);

    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( 2, 2, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    assertEquals( numCorrect, 2 );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.0001, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize_5_4_2layers_filtersize_2_4_biased_n3 ) {
    Timer timer;
    int imageSize = 5;
    int N = 3;
    int numInPlanes = 1;
    int numOutPlanes = 3;
    float data[] = {
                    1,0,1,0,1,
                    0,1,0,1,0,
                    1,0,1,0,1,
                    0,1,0,1,0,    
                    1,0,1,0,1,

                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,

                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
};
    int inputSize = imageSize * imageSize * numInPlanes * N;
    for( int i = 0; i < inputSize; i++ ) {
        data[i] -= 0.5f;
    }
    int labels[] = { 0, 1, 2 };
    int resultsSize = numOutPlanes * N;
    float *expectedResults = new float[resultsSize];
    for( int n = 0; n < N; n++ ) {
        for( int plane = 0; plane < numOutPlanes; plane++ ) {
            expectedResults[ n * numOutPlanes + plane] = -0.5;
        }
        expectedResults[ n * numOutPlanes + labels[n]] = +0.5;
    }
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(5) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(2)->biased() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(4)->biased() );
    net->addLayer( SquareLossMaker::instance() );;
//    net->print();
    for( int epoch = 0; epoch < 1000; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.1)
            ->batchSize(N)
            ->numExamples(N)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        if( epoch % 100 == 0 ) cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
//        net->print();
//        float const*results = net->getResults();
//        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << N << endl;
    assertEquals( numCorrect, N );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.01, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize_5_4_2layers_filtersize_2_4_biased_n6 ) {
    Timer timer;
    int imageSize = 5;
    int N = 6;
    int numInPlanes = 1;
    int numOutPlanes = 3;
    float data[] = {
                    1,0,1,0,1,
                    0,1,0,1,0,
                    1,0,1,0,1,
                    0,1,0,1,0,    
                    1,0,1,0,1,

                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,

                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,

                    0,1,0,1,0,
                    1,0,1,0,1,
                    0,1,0,1,0,    
                    1,0,1,0,1,
                    0,1,0,1,0,    

                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,

                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
};
    int inputSize = imageSize * imageSize * numInPlanes * N;
    for( int i = 0; i < inputSize; i++ ) {
        data[i] -= 0.5f;
    }
    int labels[] = { 0, 1, 2, 0, 1, 2 };
    int resultsSize = numOutPlanes * N;
    float *expectedResults = new float[resultsSize];
    for( int n = 0; n < N; n++ ) {
        for( int plane = 0; plane < numOutPlanes; plane++ ) {
            expectedResults[ n * numOutPlanes + plane] = -0.5;
        }
        expectedResults[ n * numOutPlanes + labels[n]] = +0.5;
    }
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(5) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(2)->biased() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(4)->biased() );
    net->addLayer( SquareLossMaker::instance() );;
//    net->print();
float weights1[] = {-0.69664, 0.58017, 0.140447, -0.205859, 0.0198638, 0.0110593, -0.388923, -0.844424, -0.472903, 0.453888, -0.616155, -0.454998};
float weights2[] = {0.207138, -0.106497, -0.1228, -0.162173, 0.1822, -0.100027, 0.0447708, 0.165723, -0.0147989, 0.109204, -0.0334504, 0.00452646, 0.198443, -0.23725, 0.105671, 0.192242, -0.0268933, 0.150674, 0.160054, -0.116846, 0.222009, 
0.226935, 0.113873, -0.153742, 0.0273874, -0.216493, 0.177896, 0.155068, -0.0809009, 0.0305763, 0.198926, -0.115796, -0.179839, -0.133567, -0.0386595, -0.166771, -0.11833, -0.219205, -0.0115777, 0.122457, 0.0984342, 
0.0616336, 0.130647, 0.192949, 0.143467, -0.130633, -0.221122, -0.154317, 0.11901, 0.00502961, 0.213079, -0.0373076, -0.0461127, -0.156646, -0.148074, -0.105763, -0.140191, 0.136911, -0.217382, 0.17574, -0.0312263, 
0.0931478, 0.0789604, -0.00794073, -0.218235, 0.0418423, 0.234828, 0.225359, -0.191966, 0.241517, 0.182442, -0.216337, -0.228462, -0.140195, 0.0493267, 0.0383108, -0.0124946, -0.093023, 0.0322872, 0.0855678, -0.0466207, 
-0.025329, -0.198314, -0.0189797, 0.147109, -0.200046, 0.20127, 0.169828, -0.173335, -0.100567, -0.195165, -0.0657755, -0.224493, -0.208405, 0.154131, 0.12547, -0.161635, -0.248707, 0.13305, -0.00289013, 0.228017, 
0.0528438, 0.0157539, 0.161637, -0.199882, 0.171727, 0.171146, -0.237469, -0.226088, 0.2026, -0.131614, 0.0631847, -0.0949208, -0.137853, -0.177839, -0.237589, -0.229862, 0.202094, 0.0531539, -0.0467284, 0.125544, 
-0.0750956, 0.225228, 0.255915, 0.076901, -0.0596187, 0.16937, -0.104811, -0.0815879, -0.196806, 0.0526821, 0.136622, -0.12163, 0.170657, -0.0956968, -0.00985565, 0.0455411, 0.0242914, 0.107953, -0.0594324, 0.124928, 
0.0875922, -0.100952, 0.155045};
float biasWeights1[] = {0.0998941, -0.365008, 0.188937};
float biasWeights2[] = {0.232961, 0.141537, 0.159074};
    net->initWeights(1, weights1, biasWeights1 );
    net->initWeights(2, weights2, biasWeights2 );
    for( int epoch = 0; epoch < 500; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.04)
            ->batchSize(N)
            ->numExamples(N)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        if( epoch % 100 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
//        net->print();
//           net->printWeightsAsCode();
//            net->printBiasWeightsAsCode();
        }
//        float const*results = net->getResults();
//        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << N << endl;
    assertEquals( numCorrect, N );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.00001, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize_5_3_2layers_filtersize_3_3_biased_n6 ) {
    Timer timer;
    int imageSize = 5;
    int N = 6;
    int numInPlanes = 1;
    int numOutPlanes = 3;
    float data[] = {
                    1,0,1,0,1,
                    0,1,0,1,0,
                    1,0,1,0,1,
                    0,1,0,1,0,    
                    1,0,1,0,1,

                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,

                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,

                    0,1,0,1,0,
                    1,0,1,0,1,
                    0,1,0,1,0,    
                    1,0,1,0,1,
                    0,1,0,1,0,    

                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,

                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
};
    int inputSize = imageSize * imageSize * numInPlanes * N;
    for( int i = 0; i < inputSize; i++ ) {
        data[i] -= 0.5f;
    }
    int labels[] = { 0, 1, 2, 0, 1, 2 };
    int resultsSize = numOutPlanes * N;
    float *expectedResults = new float[resultsSize];
    for( int n = 0; n < N; n++ ) {
        for( int plane = 0; plane < numOutPlanes; plane++ ) {
            expectedResults[ n * numOutPlanes + plane] = -0.5;
        }
        expectedResults[ n * numOutPlanes + labels[n]] = +0.5;
    }
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(5) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
    net->addLayer( SquareLossMaker::instance() );;
//    net->print();
float weights1[] = {-0.171255, 0.374466, -0.224289, -0.196481, 0.162787, 0.418841, 0.230909, 0.23731, -0.244594, -0.469993, 0.221895, -0.0145731, 0.163359, 0.276707, -0.533498, -0.376532, 0.275129, -0.298299, -0.162541, -0.497442, 0.0331104, 
0.140816, 0.339377, -0.466528, -0.260578, -0.373026, -0.0151962};
float weights2[] = {0.11266, 0.199489, 0.193306, -0.0574513, 0.266716, -0.271093, 0.0622974, 0.276959, 0.234103, -0.0329131, 0.111828, 0.255213, 0.0546736, -0.14267, -0.195783, 0.140402, -0.225388, 0.143696, 0.00776717, -0.216402, 0.13755, 
-0.0404622, 0.321655, -0.218655, -0.140874, 0.0361279, 0.227149, -0.0224601, -0.0438027, 0.0945921, 0.264248, -0.212632, 0.125262, 0.303234, 0.265334, 0.0165108, -0.119786, 0.0967013, -0.316602, 0.0735333, -0.298583, 
-0.131285, 0.158645, 0.0816884, 0.0191159, 0.233569, -0.0288674, 0.166787, 0.0839494, -0.232928, 0.32289, 0.259277, 0.28396, 0.0585126, 0.0419515, -0.315813, 0.32489, -0.208887, -0.157422, 0.223066, 0.235666, 
-0.286893, -0.00949466, -0.0232266, 0.000597281, -0.28573, 0.23746, -0.12194, 0.211189, 0.114797, 0.334012, 0.195305, 0.0269026, 0.191523, -0.0801473, 0.323508, 0.214993, -0.0651319, 0.268872, -0.270865, 0.0842015
};
float biasWeights1[] = {0.224118, -0.246188, -0.22282};
float biasWeights2[] = {-0.0863176, -0.227985, -0.147554};
    net->initWeights(1, weights1, biasWeights1 );
    net->initWeights(2, weights2, biasWeights2 );
    for( int epoch = 0; epoch < 300; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.04)
            ->batchSize(N)
            ->numExamples(N)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        if( epoch % 100 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
//        net->print();
//           net->printWeightsAsCode();
//            net->printBiasWeightsAsCode();
        }
//        float const*results = net->getResults();
//        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << N << endl;
    assertEquals( numCorrect, N );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.1, loss );

    delete net;
}

TEST( testsimpleconvolvenet, imagesize_5_3_2layers_filtersize_3_3_biased_n18 ) {
    Timer timer;
    int imageSize = 5;
    int N = 18;
    int numInPlanes = 1;
    int numOutPlanes = 3;
    float data[] = {
                    1,0,1,0,1,
                    0,1,0,1,0,
                    1,0,1,0,1,
                    0,1,0,1,0,    
                    1,0,1,0,1,
//1
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
//2
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
//3
                    0,1,0,1,0,
                    1,0,1,0,1,
                    0,1,0,1,0,    
                    1,0,1,0,1,
                    0,1,0,1,0,    
//4
                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
//5
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
//6
                    1,0,1,0,1,
                    0,1,0,1,0,
                    1,0,1,0,1,
                    0,0,0,0,0,
                    0,0,0,0,0,
//7
                    1,0,1,0,1,
                    1,0,1,0,1,
                    1,0,1,0,1,
                    0,0,0,0,0,
                    0,0,0,0,0,
//8
                    1,1,1,1,1,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,
                    0,0,0,0,0,
//9
                    0,0,0,0,0,
                    0,0,0,0,0,
                    0,0,0,1,0,    
                    0,0,1,0,1,
                    0,0,0,1,0,    
//10
                    0,0,0,0,0,
                    0,0,0,0,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
                    0,1,0,1,0,
//11
                    0,0,0,0,0,
                    0,0,0,0,0,
                    0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0,0,

//12
                    0,0,1,0,1,
                    0,0,0,1,0,
                    0,0,1,0,1,
                    0,0,0,1,0,    
                    0,0,1,0,1,
//13
                    0,0,1,0,1,
                    0,0,1,0,1,
                    0,0,1,0,1,
                    0,0,1,0,1,
                    0,0,1,0,1,
//14
                    0,0,1,1,1,
                    0,0,0,0,0,
                    0,0,1,1,1,
                    0,0,0,0,0,
                    0,0,1,1,1,
//15
                    0,1,0,0,0,
                    1,0,1,0,0,
                    0,1,0,0,0,    
                    1,0,1,0,0,
                    0,1,0,0,0,    
//16
                    0,1,0,0,0,
                    0,1,0,0,0,
                    0,1,0,0,0,
                    0,1,0,0,0,
                    0,1,0,0,0,
//17
                    0,0,0,0,0,
                    1,1,1,0,0,
                    0,0,0,0,0,
                    1,1,1,0,0,
                    0,0,0,0,0,
};
    int inputSize = imageSize * imageSize * numInPlanes * N;
    for( int i = 0; i < inputSize; i++ ) {
        data[i] -= 0.5f;
    }
    int labels[] = { 0, 1, 2, 0, 1, 2,
                    0, 1, 2, 0, 1, 2,
                    0, 1, 2, 0, 1, 2 };
    int resultsSize = numOutPlanes * N;
    float *expectedResults = new float[resultsSize];
    for( int n = 0; n < N; n++ ) {
        for( int plane = 0; plane < numOutPlanes; plane++ ) {
            expectedResults[ n * numOutPlanes + plane] = -0.5;
        }
        expectedResults[ n * numOutPlanes + labels[n]] = +0.5;
    }
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(5) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
    net->addLayer( SquareLossMaker::instance() );;
//    net->print();
    for( int epoch = 0; epoch < 3000; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.02)
            ->batchSize(N)
            ->numExamples(N)
            ->inputData(data)
            ->expectedOutputs(expectedResults)
            ->run();
        if( epoch % 100 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
//        net->print();
//           net->printWeightsAsCode();
//            net->printBiasWeightsAsCode();
        }
//        float const*results = net->getResults();
//        AccuracyHelper::printAccuracy( 2, 2, labels, results );
    }
    net->print();
    cout << "loss, E, " << net->calcLoss(expectedResults) << endl;
    float const*results = net->getResults();
    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, results );

    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getResults() );
    cout << "accuracy: " << numCorrect << "/" << N << endl;
    assertEquals( numCorrect, N );

    float loss = net->calcLoss(expectedResults);
    cout << "loss, E, " << loss << endl;
    assertLessThan( 0.1, loss );

    delete net;
}

