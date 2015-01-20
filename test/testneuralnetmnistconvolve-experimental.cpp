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

void loadMnist( string mnistDir, string setName, int *p_N, int *p_boardSize, float ****p_images, int **p_labels, float **p_expectedOutputs ) {
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

    // expected results
    *p_expectedOutputs = new float[10 * Nboards];
    for( int n = 0; n < Nlabels; n++ ) {
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
        int thisnumright = AccuracyHelper::calcNumRight( thisBatchSize, 10, &(labels[batchStart]), results );
//        cout << name << " batch " << batch << ": numright " << thisnumright << "/" << batchSize << endl;
        testNumRight += thisnumright;
    }
//    cout << "boards interval: " << ( &(boards[1][0][0]) - &(boards[0][0][0])) << endl;
//    cout << "labels interval: " << ( &(labels[1]) - &(labels[0])) << endl;
    cout << name << " overall: " << testNumRight << "/" << N << " " << ( testNumRight * 100.0f / N ) << "%" << endl;
}

void go(Config config) {
    Timer timer;

    int boardSize;

    float ***boardsFloat = 0;
    int *labels = 0;
    float *expectedOutputs = 0;

    float ***boardsTest = 0;
    int *labelsTest = 0;
    float *expectedOutputsTest = 0;
    {
        int N;
        loadMnist( config.dataDir, config.trainSet, &N, &boardSize, &boardsFloat, &labels, &expectedOutputs );

        int Ntest;
        loadMnist( config.dataDir, config.testSet, &Ntest, &boardSize, &boardsTest, &labelsTest, &expectedOutputsTest );

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
//    net->convolutionalMaker()->numFilters(32)->filterSize(5)->rel()->biased()->insert();
    for( int i = 0; i < config.numLayers; i++ ) {
//        cout << "adding convolutional layer" << endl;
        net->convolutionalMaker()->numFilters(config.numFilters)->filterSize(config.filterSize)->relu()->biased()->padZeros(config.padZeros)->insert();
    }
    net->convolutionalMaker()->numFilters(10)->filterSize(net->layers[net->layers.size()-1]->getOutputBoardSize())->tanh()->biased(config.biased)->insert();
    net->squareLossMaker()->insert();

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
            ->expectedOutputs(expectedOutputs)
            //->run();
            ->runWithCalcTrainingAccuracy(labels, &trainNumRight );
        StatefulTimer::dump(true);
        cout << "       loss L: " << loss << endl;
//        int trainNumRight = 0;
        timer.timeCheck("after epoch " + toString(epoch) );
//        net->print();
        std::cout << "train accuracy: " << trainNumRight << "/" << numToTrain << " " << (trainNumRight * 100.0f/ numToTrain) << "%" << std::endl;
        //printAccuracy( "train", net, boardsFloat, labels, batchSize, config.numTrain );
        printAccuracy( "test", net, boardsTest, labelsTest, batchSize, config.numTest );
        timer.timeCheck("after tests");
        if( config.restartable ) {
            WeightsPersister::persistWeights( config.restartableFilename, net );
        }
    }
    //float const*results = net->getResults( net->getNumLayers() - 1 );

    printAccuracy( "test", net, boardsTest, labelsTest, batchSize, config.numTest );
//    printAccuracy( "train", net, boardsFloat, labels, batchSize, config.numTrain );
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
        totalNumRight += AccuracyHelper::calcNumRight( thisBatchSize, 10, &(labelsTest[batchStart]), resultsTest );
        if( config.restartable ) {
            WeightsPersister::persistWeights( config.restartableFilename, net );
        }
    }
    cout << "test accuracy : " << totalNumRight << "/" << totalNumber << endl;

    delete net;

    delete[] expectedOutputsTest;
    delete[] labelsTest;
//    BoardsHelper::deleteBoards( &boardsTest, Ntest, boardSize );

    delete[] expectedOutputs;
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
       }
    }
    go( config );
}


