//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>
using namespace std;

#include "BoardHelper.h"
#include "MnistLoader.h"
#include "BoardPng.h"
#include "utils/Timer.h"
#include "NeuralNet.h"

void checkAccuracy( int numImages, int numNeurons, int const*labels, float const*results ) {
    int correct = 0;
    for( int n = 0; n < numImages; n++ ) {
        double maxValue = -100000;
        int bestIndex = -1;
        for( int i = 0; i < numNeurons; i++ ) {
            if( results[ n * numNeurons + i ] > maxValue ) {
                bestIndex = i;
                maxValue = results[ n * numNeurons + i ];
            }
        }
//        cout << "expected: " << labels[n] << " got " << bestIndex << endl;
        if( bestIndex == labels[n] ) {
            correct++;
        }
    }
    cout << " accuracy: " << correct << "/" << numImages << endl;
}

int main( int argc, char *argv[] ) {
    Timer timer;

    int boardSize;
    int N;
    // images
    int ***boards = MnistLoader::loadImages( "/norep/Downloads/data/mnist", "train", &N, &boardSize );
    int *labels = MnistLoader::loadLabels( "/norep/Downloads/data/mnist", "train", &N );
    timer.timeCheck("after load images");
    float ***boardsFloat = BoardsHelper::allocateBoardsFloats( N, boardSize );
    BoardsHelper::copyBoards( boardsFloat, boards, N, boardSize );
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

    int *labels2 = new int[4];
    labels2[0] = 0;
    labels2[1] = 1;
    labels2[2] = 2;
    labels2[3] = 3;
    float *expectedOutputs2 = new float[4*4];
    for( int n = 0; n < 4; n++ ) {
       int thislabel = labels2[n];
       for( int i = 0; i < 4; i++ ) {
          expectedOutputs2[n*4+i] = -1;
       }
       expectedOutputs2[n*4 + thislabel] = +1;
    }

    int numToTrain = 10000;
    NeuralNet *net = new NeuralNet(1, boardSize );
    net->addFullyConnected( 10, 1 );
    for( int epoch = 0; epoch < 100; epoch++ ) {
        net->doEpoch( 0.1, numToTrain, numToTrain, &(boardsFloat[0][0][0]), expectedOutputs );
        cout << "loss: " << net->layers[1]->calcLoss( expectedOutputs ) << endl;
        float const*results = net->layers[1]->getResults();
        checkAccuracy( numToTrain, 10, labels, results );
    }
    float const*results = net->getResults( net->getNumLayers() - 1 );

//    BoardPng::writeBoardsToPng( "testneuralnetmnist-output.png", net->layers[1]->results, 4, boardSize );
    BoardPng::writeBoardsToPng( "testneuralnetmnist-input.png", net->layers[0]->results, 4, boardSize );
    BoardPng::writeBoardsToPng( "testneuralnetmnist-weights.png", net->layers[1]->weights, 16, boardSize );

    delete net;
    delete[] expectedOutputs;
    delete[] labels2;
    delete[] labels;
    BoardsHelper::deleteBoards( &boardsFloat, N, boardSize );
    BoardsHelper::deleteBoards( &boards, N, boardSize );

    return 0;
}



