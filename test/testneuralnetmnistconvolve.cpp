//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "BoardHelper.h"
#include "MnistLoader.h"
#include "BoardPng.h"
#include "Timer.h"
#include "NeuralNet.h"
#include "test/AccuracyHelper.h"

int main( int argc, char *argv[] ) {
    Timer timer;

    int boardSize;
    int N;
    // images
    int ***boards = MnistLoader::loadImages( "/norep/Downloads/data/mnist", "train", &N, &boardSize );
    int *labels = MnistLoader::loadLabels( "/norep/Downloads/data/mnist", "train", &N );
    timer.timeCheck("after load images");
    float ***boardsFloat = BoardsHelper::allocateBoardsFloats( N, boardSize + 1 );
    BoardsHelper::copyBoards( boardsFloat, boards, N, boardSize );
    boardSize++;
    // normalize mean of each board
    for( int n = 0; n < N; n++ ) {
       float sum = 0;
       float count = 0;
       float thismax = 0;
       for( int i = 0; i < boardSize; i++ ) {
          for( int j = 0; j < boardSize; j++ ) {
              count++;
              sum += boardsFloat[n][i][j];
              thismax = max( thismax, boardsFloat[n][i][j] );
          }
       }
       float mean = sum / count;
//       cout << "mean " << mean << endl;
       for( int i = 0; i < boardSize; i++ ) {
          for( int j = 0; j < boardSize; j++ ) {
              boardsFloat[n][i][j] = boardsFloat[n][i][j] / thismax - 0.1;
          }
       }       
    }
    // expected results
    float *expectedOutputs = new float[10 * N];
    for( int n = 0; n < N; n++ ) {
       int thislabel = labels[n];
       for( int i = 0; i < 10; i++ ) {
          expectedOutputs[n*10+i] = -1;
       }
       expectedOutputs[n*10+thislabel] = +1;
    }

    int *labels2 = new int[2];
    labels2[0] = 0;
    labels2[1] = 1;
//    labels2[2] = 2;
//    labels2[3] = 3;
    float *expectedOutputs2 = new float[2 * 2];
    for( int n = 0; n < 2; n++ ) {
       int thislabel = labels2[n];
       for( int i = 0; i < 2; i++ ) {
          expectedOutputs2[n*2+i] = -1;
       }
       expectedOutputs2[n*2 + thislabel] = +1;
    }

    int numToTrain = 2;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(boardSize)->instance();
    net->convolutionalMaker()->numFilters(2)->filterSize(boardSize)->insert();
    net->print();
    for( int epoch = 0; epoch < 40; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.1)
            ->batchSize(2)
            ->numExamples(2)
            ->inputData(&(boardsFloat[0][0][0]))
            ->expectedOutputs(expectedOutputs2)
            ->run();
//        net->setBatchSize(2);
//        net->propagate(0, 2, &(boardsFloat[0][0][0]) );
////    std::cout << "************* propagate done" << std::endl;
////    net->print();
//        net->backProp( 0.001, expectedOutputs2 );
//    std::cout << "************* backpropdone" << std::endl;
//    net->print();
//        net->doEpoch( 0.001, numToTrain, numToTrain, &(boardsFloat[0][0][0]), expectedOutputs2 );
        cout << "loss: " << net->calcLoss( expectedOutputs2 ) << endl;
        float const*results = net->getResults();
        AccuracyHelper::printAccuracy( numToTrain, 2, labels2, results );
    }
    float const*results = net->getResults( net->getNumLayers() - 1 );

//    BoardPng::writeBoardsToPng( "testneuralnetmnist-output.png", net->layers[1]->results, 4, boardSize );
    BoardPng::writeBoardsToPng( "testneuralnetmnistconvolve-output0.png", net->layers[0]->results, 2, boardSize );
    BoardPng::writeBoardsToPng( "testneuralnetmnistconvolve-output1.png", net->layers[1]->results, 4, boardSize );
    BoardPng::writeBoardsToPng( "testneuralnetmnistconvolve-weights1.png", net->layers[1]->weights, 4, boardSize );

    delete net;
    delete[] expectedOutputs;
    delete[] labels2;
    delete[] labels;
    BoardsHelper::deleteBoards( &boardsFloat, N, boardSize );
    BoardsHelper::deleteBoards( &boards, N, boardSize );

    return 0;
}



