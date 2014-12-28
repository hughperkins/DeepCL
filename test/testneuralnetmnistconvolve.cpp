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
#include "stringhelper.h"

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

class Config {
public:
    string dataDir = "/norep/Downloads/data/mnist";
    string trainSet = "train";
    string testSet = "t10k";
    int numTrain = 1000;
    int numTest = 1000;
    int batchSize = 100;
    int numEpochs = 100;
    float learningRate = 0.1f;
    int biased = 1;
    Config() {
    }
};

void go(Config config) {
    Timer timer;

    int boardSize;
    int N;

    float ***boardsFloat = 0;
    int *labels = 0;
    float *expectedOutputs = 0;
    loadMnist( config.dataDir, config.trainSet, &N, &boardSize, &boardsFloat, &labels, &expectedOutputs );

    float mean;
    float thismax;
    getStats( boardsFloat, N, boardSize, &mean, &thismax );
    normalize( boardsFloat, N, boardSize, mean, thismax );

    int Ntest;
    float ***boardsTest = 0;
    int *labelsTest = 0;
    float *expectedOutputsTest = 0;
    loadMnist( config.dataDir, config.testSet, &Ntest, &boardSize, &boardsTest, &labelsTest, &expectedOutputsTest );
    normalize( boardsTest, Ntest, boardSize, mean, thismax );

    timer.timeCheck("after load images");

    int numToTrain = config.numTrain;
    int batchSize = config.batchSize;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(boardSize)->instance();
    net->convolutionalMaker()->numFilters(10)->filterSize(boardSize)->tanh()->biased(config.biased)->insert();
    net->print();
    for( int epoch = 0; epoch < config.numEpochs; epoch++ ) {
        float loss = net->epochMaker()
            ->learningRate(config.learningRate)
            ->batchSize(batchSize)
            ->numExamples(numToTrain)
            ->inputData(&(boardsFloat[0][0][0]))
            ->expectedOutputs(expectedOutputs)
            ->run();
        cout << "loss: " << loss << endl;
        int trainNumRight = 0;
        timer.timeCheck("after epoch");
        for( int batch = 0; batch < numToTrain / batchSize; batch++ ) {
            int batchStart = batch * batchSize;
            net->propagate( &(boardsFloat[batchStart][0][0]) );
            float const*results = net->getResults();
            trainNumRight += AccuracyHelper::calcNumRight( config.batchSize, 10, &(labels[batchStart]), results );
        }
        timer.timeCheck("after calc accuracy");
        cout << "train: " << trainNumRight << "/" << numToTrain << endl;
        cout << "test:" << endl;
        net->propagate( &(boardsTest[0][0][0]) );
        float const*resultsTest = net->getResults();
        AccuracyHelper::printAccuracy( config.batchSize, 10, labelsTest, resultsTest );
    }
    //float const*results = net->getResults( net->getNumLayers() - 1 );

    int numBatches = config.numTest / config.batchSize;
    int totalNumber = 0;
    int totalNumRight = 0;
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * config.batchSize;
        net->propagate( &(boardsTest[batchStart][0][0]) );
        float const*resultsTest = net->getResults();
        totalNumber += config.batchSize;
        totalNumRight += AccuracyHelper::calcNumRight( config.batchSize, 10, &(labelsTest[batchStart]), resultsTest );
    }
    cout << "test accuracy : " << totalNumRight << "/" << totalNumber << endl;

    delete net;

    delete[] expectedOutputsTest;
    delete[] labelsTest;
    BoardsHelper::deleteBoards( &boardsTest, Ntest, boardSize );

    delete[] expectedOutputs;
    delete[] labels;
    BoardsHelper::deleteBoards( &boardsFloat, N, boardSize );
}

int main( int argc, char *argv[] ) {
    Config config;
    if( argc == 2 && ( string(argv[1]) == "--help" || string(argv[1]) == "--?" || string(argv[1]) == "-?" || string(argv[1]) == "-h" ) ) {
        cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
        cout << "Possible key=value pairs:" << endl;
        cout << "    datadir=[data directory] (" << config.dataDir << ")" << endl;
        cout << "    trainset=[train|t10k|other set name] (" << config.trainSet << ")" << endl;
        cout << "    testset=[train|t10k|other set name] (" << config.testSet << ")" << endl;
        cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
        cout << "    numtest=[num test examples] (" << config.numTest << ")" << endl;
        cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
        cout << "    numepochs=[number epochs] (" << config.numEpochs << ")" << endl;
        cout << "    biased=[0|1] (" << config.biased << ")" << endl;
        cout << "    learningrate=[learning rate, a float value] (" << config.learningRate << ")" << endl;
    } 
    for( int i = 1; i < argc; i++ ) {
       vector<string> splitkeyval = split( argv[i], "=" );
       if( splitkeyval.size() != 2 ) {
          cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
          exit(1);
       } else {
           string key = splitkeyval[0];
           string value = splitkeyval[1];
           if( key == "datadir" ) config.dataDir = value;
           if( key == "trainset" ) config.trainSet = value;
           if( key == "testset" ) config.testSet = value;
           if( key == "numtrain" ) config.numTrain = atoi(value);
           if( key == "numtest" ) config.numTest = atoi(value);
           if( key == "batchsize" ) config.batchSize = atoi(value);
           if( key == "numepochs" ) config.numEpochs = atoi(value);
           if( key == "biased" ) config.biased = atoi(value);
           if( key == "learningrate" ) config.learningRate = atof(value);
       }
    }
    go( config );
}


