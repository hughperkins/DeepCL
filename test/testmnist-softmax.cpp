//#include "OpenCLHelper.h"
//#include "ClConvolve.h"

#include <iostream>

#include "BoardHelper.h"
#include "MnistLoader.h"
#include "BoardPng.h"
#include "Timer.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "stringhelper.h"
#include "FileHelper.h"
#include "StatefulTimer.h"
#include "WeightsPersister.h"

using namespace std;

void loadMnist( string mnistDir, string setName, int *p_N, int *p_boardSize, float ****p_images, int **p_labels ) {
    int boardSize;
    int Nboards;
    int Nlabels;
    // images
    int ***boards = MnistLoader::loadImages( mnistDir, setName, &Nboards, &boardSize );
    int *labels = MnistLoader::loadLabels( mnistDir, setName, &Nlabels );
    if( Nboards != Nlabels ) {
         throw runtime_error("mismatch between number of boards, and number of labels " + toString(Nboards ) + " vs " +
             toString(Nlabels ) );
    }
    cout << "loaded " << Nboards << " boards.  " << endl;
//    MnistLoader::shuffle( boards, labels, Nboards, boardSize );
    float ***boardsFloat = BoardsHelper::allocateBoardsFloats( Nboards, boardSize );
    BoardsHelper::copyBoards( boardsFloat, boards, Nboards, boardSize );
    BoardsHelper::deleteBoards( &boards, Nboards, boardSize );
    *p_images = boardsFloat;
    *p_labels = labels;
   
    *p_boardSize = boardSize;
    *p_N = Nboards;
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
    string testSet = "t10k";
    int numTrain = 60000;
    int numTest = 10000;
    int batchSize = 128;
    int numEpochs = 20;
    int numFilters = 16;
    int numLayers = 1;
    int padZeros = 0;
    int filterSize = 5;
    int restartable = 0;
    string restartableFilename = "weights.dat";
    float learningRate = 0.0001f;
    int biased = 1;
    string resultsFilename = "results.txt";
    Config() {
    }
};

void printAccuracy( string name, NeuralNet *net, float ***boards, int *labels, int batchSize, int N ) {
    int testNumRight = 0;
    net->setBatchSize( batchSize );
    int numBatches = (N + batchSize - 1 ) / batchSize;
    for( int batch = 0; batch < numBatches; batch++ ) {
        int batchStart = batch * batchSize;
        int thisBatchSize = batchSize;
        if( batch == numBatches - 1 ) {
            thisBatchSize = N - batchStart;
            net->setBatchSize( thisBatchSize );
        }
        net->propagate( &(boards[batchStart][0][0]) );
        float const*results = net->getResults();
        int thisnumright = net->calcNumRight( &(labels[batchStart]) );
        testNumRight += thisnumright;
    }
    cout << name << " overall: " << testNumRight << "/" << N << " " << ( testNumRight * 100.0f / N ) << "%" << endl;
}

void go(Config config) {
    Timer timer;

    int boardSize;

    float ***boardsFloat = 0;
    int *labels = 0;

    float ***boardsTest = 0;
    int *labelsTest = 0;
    {
        int N;
        loadMnist( config.dataDir, config.trainSet, &N, &boardSize, &boardsFloat, &labels );

        int Ntest;
        loadMnist( config.dataDir, config.testSet, &Ntest, &boardSize, &boardsTest, &labelsTest );

    }

    float mean;
    float thismax;
//    getStats( boardsFloat, config.numTrain, boardSize, &mean, &thismax );
    mean = 33;
    thismax = 255;
    cout << " board stats mean " << mean << " max " << thismax << " boardSize " << boardSize << endl;
    normalize( boardsFloat, config.numTrain, boardSize, mean, thismax );
    normalize( boardsTest, config.numTest, boardSize, mean, thismax );
    timer.timeCheck("after load images");

    int numToTrain = config.numTrain;
    const int batchSize = config.batchSize;
    NeuralNet *net = NeuralNet::maker()->planes(1)->boardSize(boardSize)->instance();
    for( int i = 0; i < config.numLayers; i++ ) {
        net->convolutionalMaker()->numFilters(config.numFilters)->filterSize(config.filterSize)->relu()->biased()->padZeros(config.padZeros)->insert();
    }
    net->fullyConnectedMaker()->numPlanes(10)->boardSize(1)->linear()->biased(config.biased)->insert();
    net->softMaxLossMaker()->insert();
    net->setBatchSize(config.batchSize);
    net->print();

    if( config.restartable ) {
        WeightsPersister::loadWeights( config.restartableFilename, net );
    }

    timer.timeCheck("before learning start");
    StatefulTimer::timeCheck("START");
    for( int epoch = 0; epoch < config.numEpochs; epoch++ ) {
        int trainNumRight = 0;
        float loss = net->epochMaker()
            ->learningRate(config.learningRate)
            ->batchSize(batchSize)
            ->numExamples(numToTrain)
            ->inputData(&(boardsFloat[0][0][0]))
            ->labels(labels)
            ->runFromLabels( &trainNumRight );
        StatefulTimer::dump(true);
        cout << "       loss L: " << loss << endl;
        timer.timeCheck("after epoch " + toString(epoch) );
//        net->print();
        std::cout << "train accuracy: " << trainNumRight << "/" << numToTrain << " " << (trainNumRight * 100.0f/ numToTrain) << "%" << std::endl;
        printAccuracy( "test", net, boardsTest, labelsTest, batchSize, config.numTest );
        timer.timeCheck("after tests");
        if( config.restartable ) {
            WeightsPersister::persistWeights( config.restartableFilename, net );
        }
    }

    printAccuracy( "test", net, boardsTest, labelsTest, batchSize, config.numTest );
    timer.timeCheck("after tests");

    int numTestBatches = ( config.numTest + config.batchSize - 1 ) / config.batchSize;
    int totalNumber = 0;
    int totalNumRight = 0;
    net->setBatchSize( config.batchSize );
    for( int batch = 0; batch < numTestBatches; batch++ ) {
        int batchStart = batch * config.batchSize;
        int thisBatchSize = config.batchSize;
        if( batch == numTestBatches - 1 ) {
            thisBatchSize = config.numTest - batchStart;
            net->setBatchSize( thisBatchSize );
        }
        net->propagate( &(boardsTest[batchStart][0][0]) );
        float const*resultsTest = net->getResults();
        totalNumber += thisBatchSize;
        totalNumRight += net->calcNumRight( &(labelsTest[batchStart]) );
        if( config.restartable ) {
            WeightsPersister::persistWeights( config.restartableFilename, net );
        }
    }
    cout << "test accuracy : " << totalNumRight << "/" << totalNumber << endl;

    delete net;

    delete[] labelsTest;
//    BoardsHelper::deleteBoards( &boardsTest, Ntest, boardSize );

    delete[] labels;
//    BoardsHelper::deleteBoards( &boardsFloat, N, boardSize );
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
        cout << "    numlayers=[number convolutional layers] (" << config.numLayers << ")" << endl;
        cout << "    numfilters=[number filters] (" << config.numFilters << ")" << endl;
        cout << "    filtersize=[filter size] (" << config.filterSize << ")" << endl;
        cout << "    biased=[0|1] (" << config.biased << ")" << endl;
        cout << "    padzeros=[0|1] (" << config.padZeros << ")" << endl;
        cout << "    learningrate=[learning rate, a float value] (" << config.learningRate << ")" << endl;
        cout << "    restartable=[weights are persistent?] (" << config.restartable << ")" << endl;
        cout << "    restartablefilename=[filename to store weights] (" << config.restartableFilename << ")" << endl;
        cout << "    resultsfilename=[filename to store results] (" << config.resultsFilename << ")" << endl;
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
           if( key == "numfilters" ) config.numFilters = atoi(value);
           if( key == "numlayers" ) config.numLayers = atoi(value);
           if( key == "padzeros" ) config.padZeros = atoi(value);
           if( key == "filtersize" ) config.filterSize = atoi(value);
           if( key == "learningrate" ) config.learningRate = atof(value);
           if( key == "restartable" ) config.restartable = atoi(value);
           if( key == "restartablefilename" ) config.restartableFilename = value;
           if( key == "resultsfilename" ) config.resultsFilename = value;
       }
    }
    go( config );
}


