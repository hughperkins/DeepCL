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

void loadMnist( string mnistDir, string setName, int *p_N, int *p_boardSize, float ****p_images, int **p_labels, float **p_expectedOutputs ) {
    int boardSize;
    int N;
    // images
    int ***boards = MnistLoader::loadImages( mnistDir, setName, &N, &boardSize );
    int *labels = MnistLoader::loadLabels( mnistDir, setName, &N );
    float ***boardsFloat = BoardsHelper::allocateBoardsFloats( N, boardSize + 1 );
    BoardsHelper::copyBoards( boardsFloat, boards, N, boardSize );
    BoardsHelper::deleteBoards( &boards, N, boardSize );
    *p_images = boardsFloat;
    *p_labels = labels;
   
    *p_boardSize = boardSize + 1;
    *p_N = N;

    // expected results
    *p_expectedOutputs = new float[10 * N];
    for( int n = 0; n < N; n++ ) {
       int thislabel = labels[n];
       for( int i = 0; i < 10; i++ ) {
          (*p_expectedOutputs)[n*10+i] = -0.5;
       }
       (*p_expectedOutputs)[n*10+thislabel] = +0.5;
    }
}

void getStats( float ***boards, int N, int boardSize, float *p_mean, float *p_thismax ) {
    // get mean of the dataset
    int count = 0;
    float thismax = 0;
   float sum = 0;
    for( int n = 0; n < N; n++ ) {
       for( int i = 0; i < boardSize; i++ ) {
          for( int j = 0; j < boardSize; j++ ) {
              count++;
              sum += boards[n][i][j];
              thismax = max( thismax, boards[n][i][j] );
          }
       }
    }
    *p_mean = sum / count;
    *p_thismax = thismax;
}

void normalize( float ***boards, int N, int boardSize, double mean, double thismax ) {
    for( int n = 0; n < N; n++ ) {
       for( int i = 0; i < boardSize; i++ ) {
          for( int j = 0; j < boardSize; j++ ) {
              boards[n][i][j] = boards[n][i][j] / thismax - 0.1;
          }
       }       
    }
}

int main( int argc, char *argv[] ) {
    Timer timer;

    int boardSize;
    int N;

    float ***boardsFloat = 0;
    int *labels = 0;
    float *expectedOutputs = 0;
    loadMnist( "/norep/Downloads/data/mnist", "train", &N, &boardSize, &boardsFloat, &labels, &expectedOutputs );

    float mean;
    float thismax;
    getStats( boardsFloat, N, boardSize, &mean, &thismax );
    normalize( boardsFloat, N, boardSize, mean, thismax );

    int Ntest;
    float ***boardsTest = 0;
    int *labelsTest = 0;
    float *expectedOutputsTest = 0;
    loadMnist( "/norep/Downloads/data/mnist", "t10k", &Ntest, &boardSize, &boardsTest, &labelsTest, &expectedOutputsTest );
    normalize( boardsTest, Ntest, boardSize, mean, thismax );

    timer.timeCheck("after load images");

    int numToTrain = 1000;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(boardSize)->instance();
    net->convolutionalMaker()->numFilters(10)->filterSize(boardSize)->insert();
    net->print();
    for( int epoch = 0; epoch < 20; epoch++ ) {
        net->epochMaker()
            ->learningRate(0.1)
            ->batchSize(numToTrain)
            ->numExamples(numToTrain)
            ->inputData(&(boardsFloat[0][0][0]))
            ->expectedOutputs(expectedOutputs)
            ->run();
        cout << "loss: " << net->calcLoss( expectedOutputs ) << endl;
        float const*results = net->getResults();
        AccuracyHelper::printAccuracy( numToTrain, 10, labels, results );
    }
    float const*results = net->getResults( net->getNumLayers() - 1 );

    net->propagate( 0, numToTrain, &(boardsTest[0][0][0]) );
    float const*resultsTest = net->getResults();
    cout << "test:" << endl;
    AccuracyHelper::printAccuracy( numToTrain, 10, labelsTest, resultsTest );

    delete net;

    delete[] expectedOutputsTest;
    delete[] labelsTest;
    BoardsHelper::deleteBoards( &boardsTest, Ntest, boardSize );

    delete[] expectedOutputs;
    delete[] labels;
    BoardsHelper::deleteBoards( &boardsFloat, N, boardSize );

    return 0;
}



