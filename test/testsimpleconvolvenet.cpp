//#include "EasyCL.h"
//#include "ClConvolve.h"

#include <iostream>
#include <vector>

#include "util/Timer.h"
#include "net/NeuralNet.h"
#include "AccuracyHelper.h"
#include "util/StatefulTimer.h"
#include "batch/NetLearner.h"
#include "trainers/SGD.h"
#include "layer/Layer.h"
#include "EasyCL.h"
#include "net/NeuralNetMould.h"
#include "layer/LayerMakers.h"
#include "batch/EpochMaker.h"
#include "batch/Batcher2.h"
#include "clblas/ClBlasInstance.h"

#include "test/DeepCLGtestGlobals.h"

#include "test/NetTestHelper.h"

#include "gtest/gtest.h"

using namespace std;

TEST( testsimpleconvolvenet, imagesize1_planes2_filters2_unbiased_tanh ) {
    Timer timer;
    const float learningRate = 0.1f;
    const int batchSize = 2;
    float *data = new float[batchSize];
    data[0] = 0.5f;
    data[1] = -0.5f;
    int *labels = new int[batchSize];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedOutput = new float[4];
    expectedOutput[0] = 0.5f;
    expectedOutput[1] = -0.5f;
    expectedOutput[2] = -0.5f;
    expectedOutput[3] = 0.5f;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(0) );
    net->addLayer( ActivationMaker::instance()->tanh() );
    net->addLayer( SquareLossMaker::instance() );;
    float weights1[] = {0.382147f, -1.77522f};
    net->initWeights(1, weights1);

//    BatchLearner batchLearner( net );
    SGD *sgd = SGD::instance( cl, learningRate, 0 );
    InputData inputData( net->getInputCubeSize(), data );
    ExpectedData expectedData( net->getOutputCubeSize(), expectedOutput );
    LearnBatcher2 learnBatcher( net, sgd, batchSize, batchSize,
            &inputData, &expectedData );
    for( int epoch = 0; epoch < 50; epoch++ ) {
        learnBatcher.run( epoch );
//        batchLearner.runEpochFromExpected( sgd, batchSize, batchSize, data, expectedOutput );
        if( epoch % 10 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
            float const*output = net->getOutput();
            AccuracyHelper::printAccuracy( 2, 2, labels, output );
        }
    }

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( 2, 2, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    EXPECT_EQ( numCorrect, 2 );
    EXPECT_GE( 0.03, loss );

    delete sgd;
    delete net;
    delete cl;
}

TEST( testsimpleconvolvenet, imagesize1_planes2_filters2_tanh ) {
    Timer timer;
    const float learningRate = 1.0f;
    const int batchSize = 2;
    float *data = new float[batchSize];
    data[0] = 0.5f;
    data[1] = -0.5f;
    int *labels = new int[batchSize];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedOutput = new float[4];
    expectedOutput[0] = 0.5f;
    expectedOutput[1] = -0.5f;
    expectedOutput[2] = -0.5f;
    expectedOutput[3] = 0.5f;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
    net->addLayer( ActivationMaker::instance()->tanh() );
    net->addLayer( SquareLossMaker::instance() );;
    float weights1[] = {0.382147f, -1.77522f};
    float bias1[] = {-1.00181f, 0.891056f};
    net->initWeights(1, weights1);
    net->initBias(1, bias1);


    SGD *sgd = SGD::instance( cl, learningRate, 0 );
    InputData inputData( net->getInputCubeSize(), data );
    ExpectedData expectedData( net->getOutputCubeSize(), expectedOutput );
    LearnBatcher2 learnBatcher( net, sgd, batchSize, batchSize,
            &inputData, &expectedData );
//    for( int epoch = 0; epoch < 50; epoch++ ) {
//        learnBatcher.run( epoch );
//    BatchLearner batchLearner( net );
//    SGD *sgd = SGD::instance( cl, learningRate, 0 );
    for( int epoch = 0; epoch < 30; epoch++ ) {
        learnBatcher.run( epoch );
//        batchLearner.runEpochFromExpected( sgd, batchSize, batchSize, data, expectedOutput );
        if( epoch % 10 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
            float const*output = net->getOutput();
            AccuracyHelper::printAccuracy( 2, 2, labels, output );
        }
    }

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( 2, 2, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    EXPECT_EQ( numCorrect, 2 );
    EXPECT_GE( 0.01, loss );

    delete sgd;
    delete net;
    delete cl;
}

TEST( testsimpleconvolvenet, imagesize3_n4_filtersize3_tanh ) {
    Timer timer;
    float data[] = { 0.5f, 0.5f, 0.5f,
                    -0.5f, 0.5f, 0.5f,
                    0.5f, 0.5f, 0.5f,
    
                   0.5f, 0.5f, 0.5f,
                   0.5f, -0.5f, 0.5f,
                   0.5f, 0.5f, 0.5f,

                    -0.5f, -0.5f, -0.5f,
                    -0.5f, 0.5f, -0.5f,
                    -0.5f, -0.5f, -0.5f,
    
                   -0.5f, -0.5f, -0.5f,
                   0.5f, -0.5f, -0.5f,
                   -0.5f, -0.5f, -0.5f
 };

    int *labels = new int[4];
    labels[0] = 0;
    labels[1] = 1;
    labels[2] = 0;
    labels[3] = 1;
    float *expectedOutput = new float[8];
    expectedOutput[0] = 0.5f;
    expectedOutput[1] = -0.5f;
    expectedOutput[2] = -0.5f;
    expectedOutput[3] = 0.5f;
    expectedOutput[4] = 0.5f;
    expectedOutput[5] = -0.5f;
    expectedOutput[6] = -0.5f;
    expectedOutput[7] = 0.5f;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(3) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased() );
    net->addLayer( ActivationMaker::instance()->tanh() );
    net->addLayer( SquareLossMaker::instance() );;
    float weights1[] = {-0.171115f, 0.28369f, 0.201354f, -0.496124f, 0.391512f, 0.120458f, 0.396952f, -0.1356f, -0.319595f, 0.251043f, 0.318859f, 0.220892f, -0.480651f, -0.51708f, 0.2173f, 0.365935f, 0.304687f, -0.712624f};
    float bias1[] = {0.375101f, 0.00130748f};
    net->initWeights(1, weights1);
    net->initBias(1, bias1 );
    float const*output = 0;

    SGD *sgd = SGD::instance( cl, 0.4f, 0 );
    InputData inputData( net->getInputCubeSize(), data );
    ExpectedData expectedData( net->getOutputCubeSize(), expectedOutput );
    LearnBatcher2 learnBatcher( net, sgd, 4, 4, &inputData, &expectedData );
    for( int epoch = 0; epoch < 15; epoch++ ) {
        learnBatcher.run( epoch );
//        batchLearner.runEpochFromExpected( sgd, 4, 4, data, expectedOutput );
//        net->printWeightsAsCode();
//        net->printBiasAsCode();
        if( epoch % 5 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
            output = net->getOutput();
            AccuracyHelper::printAccuracy( 4, 2, labels, output );
        }
    }
//    net->print();
    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, output );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    EXPECT_EQ( numCorrect, 4 );
    EXPECT_GE( 0.0001f, loss );
    
    delete sgd;
    delete net;
    delete cl;
}

TEST( testsimpleconvolvenet, imagesize1_2planes_filtersize1 ) {
    Timer timer;
    float *data = new float[2];
    data[0] = 0.5f;
    data[1] = -0.5f;
    int *labels = new int[2];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedOutput = new float[4];
    expectedOutput[0] = 1;
    expectedOutput[1] = 0;
    expectedOutput[2] = 0;
    expectedOutput[3] = 1;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
//    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( SquareLossMaker::instance() );
    float weights1[] = {-0.380177f, -1.5738f};
    float bias1[] = {0.5f, 0.0606055f};
    net->initWeights( 1, weights1, bias1 );

    SGD *sgd = SGD::instance( cl, 0.2f, 0 );
    InputData inputData( net->getInputCubeSize(), data );
    ExpectedData expectedData( net->getOutputCubeSize(), expectedOutput );
    LearnBatcher2 learnBatcher( net, sgd, 2, 2, &inputData, &expectedData );
    for( int epoch = 0; epoch < 40; epoch++ ) {
        learnBatcher.run( epoch );
//    BatchLearner batchLearner( net );
//    SGD *sgd = SGD::instance( cl, 0.2f, 0 );
//    for( int epoch = 0; epoch < 40; epoch++ ) {
//        batchLearner.runEpochFromExpected( sgd, 2, 2, data, expectedOutput );
        if( epoch % 5 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//            net->print();
    //        net->printWeightsAsCode();
    //        net->printBiasAsCode();
            float const*output = net->getOutput();
            AccuracyHelper::printAccuracy( 2, 2, labels, output );
        }
    }
//    net->print();
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( 2, 2, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    EXPECT_EQ( numCorrect, 2 );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.001f, loss );

    delete sgd;
    delete net;
    delete cl;
}

TEST( testsimpleconvolvenet, imagesize3_n4_filtersize3_relu ) {
    Timer timer;
    float data[] = { 0.5f, 0.5f, 0.5f,
                    -0.5f, 0.5f, 0.5f,
                    0.5f, 0.5f, 0.5f,
    
                   0.5f, 0.5f, 0.5f,
                   0.5f, -0.5f, 0.5f,
                   0.5f, 0.5f, 0.5f,

                    -0.5f, -0.5f, -0.5f,
                    -0.5f, 0.5f, -0.5f,
                    -0.5f, -0.5f, -0.5f,
    
                   -0.5f, -0.5f, -0.5f,
                   0.5f, -0.5f, -0.5f,
                   -0.5f, -0.5f, -0.5f
 };

    int *labels = new int[4];
    labels[0] = 0;
    labels[1] = 1;
    labels[2] = 0;
    labels[3] = 1;
    float *expectedOutput = new float[8];
    expectedOutput[0] = 1;
    expectedOutput[1] = 0;
    expectedOutput[2] = 0;
    expectedOutput[3] = 1;
    expectedOutput[4] = 1;
    expectedOutput[5] = 0;
    expectedOutput[6] = 0;
    expectedOutput[7] = 1;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(3) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(3)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( SquareLossMaker::instance() );
    float const*output = 0;
    double _weights1[] = {0.0113327, 0.280063, -0.0584702, -0.503431, -0.37286, -0.457257, 0.29226, -0.360089, -0.273977, 0.530185, -0.460167, 0.489126, 0.141883, 0.179525, -0.18084, 0.412117, 0.0866731, -0.247958};
    vector<float> __weights1( _weights1, _weights1 + sizeof( _weights1 ) / sizeof(double) );
    float *weights1 = &__weights1[0];
    float bias1[] = {0.0418723f, 0.158733f};
    net->getLayer(1)->setWeights( weights1, bias1 );
//    BatchLearner batchLearner( net );
    SGD *sgd = SGD::instance( cl, 0.1f, 0 );
    for( int epoch = 0; epoch < 50; epoch++ ) {
        net->epochMaker(sgd)
            ->batchSize(4)
            ->numExamples(4)
            ->inputData(data)
            ->expectedOutputs(expectedOutput)
            ->run(epoch);
        if( epoch % 5 == 0 ) {
            output = net->getOutput();
    //        net->printWeightsAsCode();
    //        net->printBiasAsCode();
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
            AccuracyHelper::printAccuracy( 4, 2, labels, output );
        }
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, output );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    EXPECT_EQ( numCorrect, 4 );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.000001, loss );

    delete sgd;
    delete net;
    delete cl;
}

TEST( testsimpleconvolvenet, imagesize3_n4_filtersize3_linear ) {
    Timer timer;
    float data[] = { 0.5f, 0.5f, 0.5f,
                    -0.5f, 0.5f, 0.5f,
                    0.5f, 0.5f, 0.5f,
    
                   0.5f, 0.5f, 0.5f,
                   0.5f, -0.5f, 0.5f,
                   0.5f, 0.5f, 0.5f,

                    -0.5f, -0.5f, -0.5f,
                    -0.5f, 0.5f, -0.5f,
                    -0.5f, -0.5f, -0.5f,
    
                   -0.5f, -0.5f, -0.5f,
                   0.5f, -0.5f, -0.5f,
                   -0.5f, -0.5f, -0.5
 };

    int *labels = new int[4];
    labels[0] = 0;
    labels[1] = 1;
    labels[2] = 0;
    labels[3] = 1;
    float *expectedOutput = new float[8];
    expectedOutput[0] = 1;
    expectedOutput[1] = 0;
    expectedOutput[2] = 0;
    expectedOutput[3] = 1;
    expectedOutput[4] = 1;
    expectedOutput[5] = 0;
    expectedOutput[6] = 0;
    expectedOutput[7] = 1;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(3) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(3)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(3)->biased() );
    net->addLayer( SquareLossMaker::instance() );;
    float const*output = 0;
    float weights1[] = {0.715867f, -0.428623f, -0.281465f, -0.736675f, -0.224507f, 0.335028f, -0.384762f, -0.213304f, 0.679177f, -0.170055f, 0.335075f, -0.572057f, -0.175718f, -0.410962f, -0.175277f, 0.536131f, -0.0568329f, -0.00297278f};
    float bias1[] = {0.5f, 0.5f};
    net->initWeights( 1, weights1, bias1 );
    SGD *sgd = SGD::instance( cl, 0.09f, 0 );
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker(sgd)
            ->batchSize(4)
            ->numExamples(4)
            ->inputData(data)
            ->expectedOutputs(expectedOutput)
            ->run(epoch);
//        net->printWeightsAsCode();
//        net->printBiasAsCode();
        if( epoch % 5 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
            output = net->getOutput();
            AccuracyHelper::printAccuracy( 4, 2, labels, output );
        }
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    AccuracyHelper::printAccuracy( 4, 2, labels, output );
    int numCorrect = AccuracyHelper::calcNumRight( 4, 2, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << 4 << endl;
    EXPECT_EQ( numCorrect, 4 );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.001f, loss );

    delete sgd;
    delete net;
    delete cl;
}

TEST( testsimpleconvolvenet, imagesize1_n2_2layers_unbiased ) {
    Timer timer;
    float *data = new float[2];
    data[0] = 0.5f;
    data[1] = -0.5f;
    int *labels = new int[2];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedOutput = new float[4];
    expectedOutput[0] = 0.5f;
    expectedOutput[1] = -0.5f;
    expectedOutput[2] = -0.5f;
    expectedOutput[3] = 0.5f;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(1) );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased(1) );
//    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( SquareLossMaker::instance() );

    float weights1[] = {-0.303866f, -1.66244f};
    float weights3[] = {0.426358f, -0.841404f, -0.420361f, 0.841048f};
    float bias1[] = {-0.324465f, 0.731219f};
    float bias3[] = {0.600115f, -0.599876f};
    net->initWeights( 1, weights1, bias1 );
    net->initWeights( 3, weights3, bias3 );

    SGD *sgd = SGD::instance( cl, 0.1f, 0.0f );
    for( int epoch = 0; epoch < 40; epoch++ ) {
        net->epochMaker(sgd)
            ->batchSize(2)
            ->numExamples(2)
            ->inputData(data)
            ->expectedOutputs(expectedOutput)
            ->run(epoch);
        cout << "epoch " << epoch << " loss, E, " << net->calcLoss(expectedOutput) << endl;
//        net->print();
//        float const*output = net->getOutput();
//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
    }
    net->print();
    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( 2, 2, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    EXPECT_EQ( numCorrect, 2 );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.0001f, loss );

        cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    net->print();
    net->getLayer(1)->getWeights();
    net->getLayer(3)->getWeights();
    NetTestHelper::printWeightsAsCode( net );
    NetTestHelper::printBiasAsCode( net );

    delete sgd;
    delete net;
    delete cl;
}

TEST( testsimpleconvolvenet, imagesize1_n2_2layers_biased ) {
    Timer timer;
    float *data = new float[2];
    data[0] = 0.5f;
    data[1] = -0.5f;
    int *labels = new int[2];
    labels[0] = 0;
    labels[1] = 1;
    float *expectedOutput = new float[4];
    expectedOutput[0] = 0.5f;
    expectedOutput[1] = -0.5f;
    expectedOutput[2] = -0.5f;
    expectedOutput[3] = 0.5f;
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(1) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(1)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(2)->filterSize(1)->biased() );
//    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( SquareLossMaker::instance() );
float weights1[] = {1.12739f, 1.21476f};
float weights2[] = {-0.352846f, 0.534554f, -1.13343f, -0.191175f};
float bias1[] = {0.971267f, 1.42629f};
float bias2[] = {-0.071288f, 0.443919f};
    net->initWeights(1, weights1, bias1 );
    net->initWeights(3, weights2, bias2 );
    SGD *sgd = SGD::instance( cl, 0.2f, 0 );
    for( int epoch = 0; epoch < 30; epoch++ ) {
        net->epochMaker(sgd)
            ->batchSize(2)
            ->numExamples(2)
            ->inputData(data)
            ->expectedOutputs(expectedOutput)
            ->run(epoch);
        if( epoch % 5 == 0 ) {
//           net->printWeightsAsCode();
//            net->printBiasAsCode();
        cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
        }
//        net->print();
//        float const*output = net->getOutput();
//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
    }
//    net->print();

    StatefulTimer::dump(true);

    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( 2, 2, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( 2, 2, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << 2 << endl;
    EXPECT_EQ( numCorrect, 2 );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.0001f, loss );

    delete sgd;
    delete net;
    delete cl;
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
    int inputNumElements = imageSize * imageSize * numInPlanes * N;
    for( int i = 0; i < inputNumElements; i++ ) {
        data[i] -= 0.5f;
    }
    int labels[] = { 0, 1, 2 };
    int outputNumElements = numOutPlanes * N;
    float *expectedOutput = new float[outputNumElements];
    for( int n = 0; n < N; n++ ) {
        for( int plane = 0; plane < numOutPlanes; plane++ ) {
            expectedOutput[ n * numOutPlanes + plane] = -0.5f;
        }
        expectedOutput[ n * numOutPlanes + labels[n]] = +0.5f;
    }
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(5) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(2)->biased() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(4)->biased() );
//    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( SquareLossMaker::instance() );
//    net->print();
    SGD *sgd = SGD::instance( cl, 0.01f, 0 );
    for( int epoch = 0; epoch < 1000; epoch++ ) {
        net->epochMaker(sgd)
            ->batchSize(N)
            ->numExamples(N)
            ->inputData(data)
            ->expectedOutputs(expectedOutput)
            ->run(epoch);
        if( epoch % 100 == 0 ) cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//        net->print();
//        float const*output = net->getOutput();
//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << N << endl;
    EXPECT_EQ( numCorrect, N );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.01f, loss );

    delete sgd;
    delete net;
    delete cl;
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
    int inputNumElements = imageSize * imageSize * numInPlanes * N;
    for( int i = 0; i < inputNumElements; i++ ) {
        data[i] -= 0.5f;
    }
    int labels[] = { 0, 1, 2, 0, 1, 2 };
    int outputNumElements = numOutPlanes * N;
    float *expectedOutput = new float[outputNumElements];
    for( int n = 0; n < N; n++ ) {
        for( int plane = 0; plane < numOutPlanes; plane++ ) {
            expectedOutput[ n * numOutPlanes + plane] = -0.5f;
        }
        expectedOutput[ n * numOutPlanes + labels[n]] = +0.5f;
    }
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(5) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(2)->biased() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(4)->biased() );
//    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( SquareLossMaker::instance() );
//    net->print();
double _weights1[] = {-0.69664, 0.58017, 0.140447, -0.205859, 0.0198638, 0.0110593, -0.388923, -0.844424, -0.472903, 0.453888, -0.616155, -0.454998};
double _weights2[] = {0.207138, -0.106497, -0.1228, -0.162173, 0.1822, -0.100027, 0.0447708, 0.165723, -0.0147989, 0.109204, -0.0334504, 0.00452646, 0.198443, -0.23725, 0.105671, 0.192242, -0.0268933, 0.150674, 0.160054, -0.116846, 0.222009, 
0.226935, 0.113873, -0.153742, 0.0273874, -0.216493, 0.177896, 0.155068, -0.0809009, 0.0305763, 0.198926, -0.115796, -0.179839, -0.133567, -0.0386595, -0.166771, -0.11833, -0.219205, -0.0115777, 0.122457, 0.0984342, 
0.0616336, 0.130647, 0.192949, 0.143467, -0.130633, -0.221122, -0.154317, 0.11901, 0.00502961, 0.213079, -0.0373076, -0.0461127, -0.156646, -0.148074, -0.105763, -0.140191, 0.136911, -0.217382, 0.17574, -0.0312263, 
0.0931478, 0.0789604, -0.00794073, -0.218235, 0.0418423, 0.234828, 0.225359, -0.191966, 0.241517, 0.182442, -0.216337, -0.228462, -0.140195, 0.0493267, 0.0383108, -0.0124946, -0.093023, 0.0322872, 0.0855678, -0.0466207, 
-0.025329, -0.198314, -0.0189797, 0.147109, -0.200046, 0.20127, 0.169828, -0.173335, -0.100567, -0.195165, -0.0657755, -0.224493, -0.208405, 0.154131, 0.12547, -0.161635, -0.248707, 0.13305, -0.00289013, 0.228017, 
0.0528438, 0.0157539, 0.161637, -0.199882, 0.171727, 0.171146, -0.237469, -0.226088, 0.2026, -0.131614, 0.0631847, -0.0949208, -0.137853, -0.177839, -0.237589, -0.229862, 0.202094, 0.0531539, -0.0467284, 0.125544, 
-0.0750956, 0.225228, 0.255915, 0.076901, -0.0596187, 0.16937, -0.104811, -0.0815879, -0.196806, 0.0526821, 0.136622, -0.12163, 0.170657, -0.0956968, -0.00985565, 0.0455411, 0.0242914, 0.107953, -0.0594324, 0.124928, 
0.0875922, -0.100952, 0.155045};
vector<float> vweights1( _weights1, _weights1 + sizeof(_weights1) / sizeof(_weights1[0] ) );
float *weights1 = &vweights1[0];
vector<float> vweights2( _weights2, _weights2 + sizeof(_weights2) / sizeof(_weights2[0] ) );
float *weights2 = &vweights2[0];
float bias1[] = {0.0998941f, -0.365008f, 0.188937f};
float bias2[] = {0.232961f, 0.141537f, 0.159074f};
    net->initWeights(1, weights1, bias1 );
    net->initWeights(3, weights2, bias2 );
    SGD *sgd = SGD::instance( cl, 0.04f, 0 );
    for( int epoch = 0; epoch < 500; epoch++ ) {
        net->epochMaker(sgd)
            ->batchSize(N)
            ->numExamples(N)
            ->inputData(data)
            ->expectedOutputs(expectedOutput)
            ->run(epoch);
        if( epoch % 100 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//        net->print();
//           net->printWeightsAsCode();
//            net->printBiasAsCode();
        }
//        float const*output = net->getOutput();
//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << N << endl;
    EXPECT_EQ( numCorrect, N );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.00001f, loss );

    delete sgd;
    delete net;
    delete cl;
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
    int inputNumElements = imageSize * imageSize * numInPlanes * N;
    for( int i = 0; i < inputNumElements; i++ ) {
        data[i] -= 0.5f;
    }
    int labels[] = { 0, 1, 2, 0, 1, 2 };
    int outputNumElements = numOutPlanes * N;
    float *expectedOutput = new float[outputNumElements];
    for( int n = 0; n < N; n++ ) {
        for( int plane = 0; plane < numOutPlanes; plane++ ) {
            expectedOutput[ n * numOutPlanes + plane] = -0.5f;
        }
        expectedOutput[ n * numOutPlanes + labels[n]] = +0.5f;
    }
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(5) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
//    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( SquareLossMaker::instance() );
//    net->print();
double _weights1[] = {-0.171255, 0.374466, -0.224289, -0.196481, 0.162787, 0.418841, 0.230909, 0.23731, -0.244594, -0.469993, 0.221895, -0.0145731, 0.163359, 0.276707, -0.533498, -0.376532, 0.275129, -0.298299, -0.162541, -0.497442, 0.0331104, 
0.140816, 0.339377, -0.466528, -0.260578, -0.373026, -0.0151962};
double _weights2[] = {0.11266, 0.199489, 0.193306, -0.0574513, 0.266716, -0.271093, 0.0622974, 0.276959, 0.234103, -0.0329131, 0.111828, 0.255213, 0.0546736, -0.14267, -0.195783, 0.140402, -0.225388, 0.143696, 0.00776717, -0.216402, 0.13755, 
-0.0404622, 0.321655, -0.218655, -0.140874, 0.0361279, 0.227149, -0.0224601, -0.0438027, 0.0945921, 0.264248, -0.212632, 0.125262, 0.303234, 0.265334, 0.0165108, -0.119786, 0.0967013, -0.316602, 0.0735333, -0.298583, 
-0.131285, 0.158645, 0.0816884, 0.0191159, 0.233569, -0.0288674, 0.166787, 0.0839494, -0.232928, 0.32289, 0.259277, 0.28396, 0.0585126, 0.0419515, -0.315813, 0.32489, -0.208887, -0.157422, 0.223066, 0.235666, 
-0.286893, -0.00949466, -0.0232266, 0.000597281, -0.28573, 0.23746, -0.12194, 0.211189, 0.114797, 0.334012, 0.195305, 0.0269026, 0.191523, -0.0801473, 0.323508, 0.214993, -0.0651319, 0.268872, -0.270865, 0.0842015
};
vector<float> __weights1( _weights1, _weights1 + sizeof( _weights1 ) / sizeof(double) );
vector<float> __weights2( _weights2, _weights2 + sizeof( _weights2  ) / sizeof(double) );
float *weights1 = &__weights1[0];
float *weights2 = &__weights2[0];
float bias1[] = {0.224118f, -0.246188f, -0.22282f};
float bias2[] = {-0.0863176f, -0.227985f, -0.147554f};
    net->initWeights(1, weights1, bias1 );
    net->initWeights(3, weights2, bias2 );
    SGD *sgd = SGD::instance( cl, 0.04f, 0 );
    for( int epoch = 0; epoch < 300; epoch++ ) {
        net->epochMaker(sgd)
            ->batchSize(N)
            ->numExamples(N)
            ->inputData(data)
            ->expectedOutputs(expectedOutput)
            ->run(epoch);
        if( epoch % 100 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//        net->print();
//           net->printWeightsAsCode();
//            net->printBiasAsCode();
        }
//        float const*output = net->getOutput();
//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
    }
//    net->print();
    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << N << endl;
    EXPECT_EQ( numCorrect, N );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.1f, loss );

    delete sgd;
    delete net;
    delete cl;
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
    int inputNumElements = imageSize * imageSize * numInPlanes * N;
    for( int i = 0; i < inputNumElements; i++ ) {
        data[i] -= 0.5f;
    }
    int labels[] = { 0, 1, 2, 0, 1, 2,
                    0, 1, 2, 0, 1, 2,
                    0, 1, 2, 0, 1, 2 };
    int outputNumElements = numOutPlanes * N;
    float *expectedOutput = new float[outputNumElements];
    for( int n = 0; n < N; n++ ) {
        for( int plane = 0; plane < numOutPlanes; plane++ ) {
            expectedOutput[ n * numOutPlanes + plane] = -0.5f;
        }
        expectedOutput[ n * numOutPlanes + labels[n]] = +0.5f;
    }
    EasyCL *cl = DeepCLGtestGlobals_createEasyCL();
    ClBlasInstance blasInstance;
    NeuralNet *net = NeuralNet::maker(cl)->instance();
    net->addLayer( InputLayerMaker::instance()->numPlanes(1)->imageSize(5) );
//    net->inputMaker<float>()->numPlanes(1)->imageSize(5)->insert();
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(3)->filterSize(3)->biased() );
//    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( SquareLossMaker::instance() );
//    net->print();
    SGD *sgd = SGD::instance( cl, 0.02f, 0 );
    for( int epoch = 0; epoch < 3000; epoch++ ) {
        net->epochMaker(sgd)
            ->batchSize(N)
            ->numExamples(N)
            ->inputData(data)
            ->expectedOutputs(expectedOutput)
            ->run(epoch);
        if( epoch % 100 == 0 ) {
            cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
//        net->print();
//           net->printWeightsAsCode();
//            net->printBiasAsCode();
        }
//        float const*output = net->getOutput();
//        AccuracyHelper::printAccuracy( 2, 2, labels, output );
    }
    net->print();
    cout << "loss, E, " << net->calcLoss(expectedOutput) << endl;
    float const*output = net->getOutput();
    AccuracyHelper::printAccuracy( N, numOutPlanes, labels, output );

    int numCorrect = AccuracyHelper::calcNumRight( N, numOutPlanes, labels, net->getOutput() );
    cout << "accuracy: " << numCorrect << "/" << N << endl;
    EXPECT_EQ( numCorrect, N );

    float loss = net->calcLoss(expectedOutput);
    cout << "loss, E, " << loss << endl;
    EXPECT_GE( 0.1f, loss );

    delete sgd;
    delete net;
    delete cl;
}

