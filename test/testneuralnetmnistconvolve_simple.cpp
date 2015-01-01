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
    string dataDir = "../data/mnist";
    string trainSet = "train";
    int numTrain = 2;
    int batchSize = 2;
    int numEpochs = 20;
    float learningRate = 1;
    int biased = 0;
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

    timer.timeCheck("after load images");

    int numToTrain = config.numTrain;
    int batchSize = config.batchSize;
    int numCategories = 10;
    if( numToTrain < 10 ) {
        numCategories = numToTrain;
        for( int n = 0; n < numToTrain; n++ ) {
           int thislabel = n;
           labels[n] = n;
           for( int i = 0; i < numCategories; i++ ) {
              expectedOutputs[n*numCategories+i] = -0.5;
           }
           expectedOutputs[n*numCategories+thislabel] = +0.5;
        }
    }

    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(boardSize)->instance();
    net->convolutionalMaker()->numFilters(numCategories)->filterSize(boardSize)->tanh()->biased(config.biased)->insert();
    net->print();
    float loss = 0;
    for( int epoch = 0; epoch < config.numEpochs; epoch++ ) {
        loss = net->epochMaker()
            ->learningRate(config.learningRate)
            ->batchSize(batchSize)
            ->numExamples(numToTrain)
            ->inputData(&(boardsFloat[0][0][0]))
            ->expectedOutputs(expectedOutputs)
            ->run();
        cout << "loss: " << loss << endl;
        int trainNumRight = 0;
        float const*results = net->getResults();
        trainNumRight += AccuracyHelper::calcNumRight( config.batchSize, numCategories, &(labels[0]), results );
        cout << "train: " << trainNumRight << "/" << numToTrain << endl;
    }
        net->print();
        cout << "loss: " << loss << endl;
        int trainNumRight = 0;
        float const*results = net->getResults();
        trainNumRight += AccuracyHelper::calcNumRight( config.batchSize, numCategories, &(labels[0]), results );
        cout << "train: " << trainNumRight << "/" << numToTrain << endl;

    delete net;

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
        cout << "    numtrain=[num training examples] (" << config.numTrain << ")" << endl;
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
           if( key == "numtrain" ) config.numTrain = atoi(value);
           if( key == "batchsize" ) config.batchSize = atoi(value);
           if( key == "numepochs" ) config.numEpochs = atoi(value);
           if( key == "biased" ) config.biased = atoi(value);
           if( key == "learningrate" ) config.learningRate = atof(value);
       }
    }
    go( config );
}


