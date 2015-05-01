//#include "EasyCL.h"
//#include "ClConvolve.h"

#include <iostream>
#ifdef MPI_AVAILABLE
#include "mpi.h"
#endif

#include "ImageHelper.h"
#include "MnistLoader.h"
// #include "ImagePng.h"
#include "util/Timer.h"
#include "net/NeuralNet.h"
#include "AccuracyHelper.h"
#include "util/stringhelper.h"
#include "util/FileHelper.h"
#include "util/StatefulTimer.h"
#include "WeightsPersister.h"

using namespace std;

int myrank = 0;
int mysize = 1;

void loadMnist( string mnistDir, string setName, int *p_N, int *p_imageSize, float ****p_images, int **p_labels ) {
    int imageSize;
    int Nimages;
    int Nlabels;
    // images
    int ***images = MnistLoader::loadImages( mnistDir, setName, &Nimages, &imageSize );
    int *labels = MnistLoader::loadLabels( mnistDir, setName, &Nlabels );
    if( Nimages != Nlabels ) {
         throw runtime_error("mismatch between number of images, and number of labels " + toString(Nimages ) + " vs " +
             toString(Nlabels ) );
    }
    if( myrank == 0 ) cout << "loaded " << Nimages << " images.  " << endl;
//    MnistLoader::shuffle( images, labels, Nimages, imageSize );
    float ***imagesFloat = ImagesHelper::allocateImagesFloats( Nimages, imageSize );
    ImagesHelper::copyImages( imagesFloat, images, Nimages, imageSize );
    ImagesHelper::deleteImages( &images, Nimages, imageSize );
    *p_images = imagesFloat;
    *p_labels = labels;
   
    *p_imageSize = imageSize;
    *p_N = Nimages;
}

void getStats( float ***images, int N, int imageSize, float *p_mean, float *p_thismax ) {
    // get mean of the dataset
    int count = 0;
    float thismax = 0;
   float sum = 0;
    for( int n = 0; n < N; n++ ) {
       for( int i = 0; i < imageSize; i++ ) {
          for( int j = 0; j < imageSize; j++ ) {
              count++;
              sum += images[n][i][j];
              thismax = max( thismax, images[n][i][j] );
          }
       }
    }
    *p_mean = sum / count;
    *p_thismax = thismax;
}

void normalize( float ***images, int N, int imageSize, double mean, double thismax ) {
    for( int n = 0; n < N; n++ ) {
       for( int i = 0; i < imageSize; i++ ) {
          for( int j = 0; j < imageSize; j++ ) {
              images[n][i][j] = images[n][i][j] / thismax - 0.1;
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
//    int restartable = 0;
//    string restartableFilename = "weights.dat";
    float learningRate = 0.0001f;
    int biased = 1;
//    string outputFilename = "output.txt";
    Config() {
    }
};

float printAccuracy( string name, NeuralNet *net, float ***images, int *labels, int batchSize, int N ) {
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
        net->forward( &(images[batchStart][0][0]) );
        float const*output = net->getOutput();
        int thisnumright = net->calcNumRight( &(labels[batchStart]) );
        testNumRight += thisnumright;
    }
    float accuracy = ( testNumRight * 100.0f / N );
    if( myrank == 0 ) cout << name << " overall: " << testNumRight << "/" << N << " " << accuracy << "%" << endl;
    return accuracy;
}

void go(Config config) {
    Timer timer;

    int imageSize;

    float ***imagesFloat = 0;
    int *labels = 0;

    float ***imagesTest = 0;
    int *labelsTest = 0;
    {
        int N;
        loadMnist( config.dataDir, config.trainSet, &N, &imageSize, &imagesFloat, &labels );

        int Ntest;
        loadMnist( config.dataDir, config.testSet, &Ntest, &imageSize, &imagesTest, &labelsTest );

    }

    float mean;
    float thismax;
//    getStats( imagesFloat, config.numTrain, imageSize, &mean, &thismax );
    mean = 33;
    thismax = 255;
    if( myrank == 0 ) cout << " image stats mean " << mean << " max " << thismax << " imageSize " << imageSize << endl;
    normalize( imagesFloat, config.numTrain, imageSize, mean, thismax );
    normalize( imagesTest, config.numTest, imageSize, mean, thismax );
    if( myrank == 0 ) timer.timeCheck("after load images");

    int numToTrain = config.numTrain;
    const int batchSize = config.batchSize;
    NeuralNet *net = NeuralNet::maker()->instance();
    net->addLayer( InputLayerMaker<float>::instance()->numPlanes(1)->imageSize(imageSize) );
    for( int i = 0; i < config.numLayers; i++ ) {
        net->addLayer( ConvolutionalMaker::instance()->numFilters(config.numFilters)->filterSize(config.filterSize)->relu()->biased()->padZeros(config.padZeros) );
    }
    net->addLayer( FullyConnectedMaker::instance()->numPlanes(10)->imageSize(1)->linear()->biased(config.biased) );
    net->addLayer( SoftMaxMaker::instance() );
    net->setBatchSize(config.batchSize);
    net->print();

//    if( config.restartable ) {
//        WeightsPersister::loadWeights( config.restartableFilename, net );
//    }

    timer.timeCheck("before learning start");
    StatefulTimer::timeCheck("START");
    const int totalWeightsSize = WeightsPersister::getTotalNumWeights(net);
    cout << "totalweightssize: " << totalWeightsSize << endl;
    float *weightsCopy = new float[totalWeightsSize];
    float *newWeights = new float[totalWeightsSize];
    float *weightsChange = new float[totalWeightsSize];
    float *weightsChangeReduced = new float[totalWeightsSize];
    for( int epoch = 0; epoch < config.numEpochs; epoch++ ) {
        int trainTotalNumber = 0;
        int trainNumRight = 0;
        int numBatches = ( config.numTrain + config.batchSize - 1 ) / config.batchSize;
        int eachNodeBatchSize = config.batchSize / mysize;
        int thisNodeBatchSize = eachNodeBatchSize;
        if( myrank == mysize - 1 ) {
            thisNodeBatchSize = config.batchSize - ( mysize - 1 ) * eachNodeBatchSize;
        }
        net->setBatchSize( thisNodeBatchSize );
        float loss = 0;
        for( int batch = 0; batch < numBatches; batch++ ) {
            int batchStart = batch * config.batchSize;
            int thisBatchSize = config.batchSize;
            int nodeBatchStart = batchStart + myrank * eachNodeBatchSize;
            if( batch == numBatches - 1 ) {
                thisBatchSize = config.numTrain - batchStart;
                eachNodeBatchSize = thisBatchSize / mysize;
                nodeBatchStart = batchStart + myrank * eachNodeBatchSize;
                thisNodeBatchSize = eachNodeBatchSize;
                if( myrank == mysize - 1 ) {
                    thisNodeBatchSize = thisBatchSize - ( mysize - 1 ) * eachNodeBatchSize;
                }
                net->setBatchSize( thisNodeBatchSize );
            }
            #ifdef MPI_AVAILABLE
            StatefulTimer::timeCheck("copyNetWeightsToArray START");
            WeightsPersister::copyNetWeightsToArray( net, weightsCopy );
            StatefulTimer::timeCheck("copyNetWeightsToArray END");
            #endif
            net->forward( &(imagesFloat[nodeBatchStart][0][0]) );
            net->backwardFromLabels( config.learningRate, &(labels[nodeBatchStart]) );
            trainTotalNumber += thisNodeBatchSize;
            trainNumRight += net->calcNumRight( &(labels[nodeBatchStart]) );
            loss += net->calcLossFromLabels( &(labels[nodeBatchStart]) );
            // share out the weights... just average them?
            // for each weight, wnew = wold + dw
            // if multiple changes, wnew = wold + dw1 + dw2 + dw3 + ...
            // if each node doing it, then wnew1 = wold + dw1; wnew2 = wold + dw2
            // wnew1 + wnew2 = wold * 2 + dw1 + dw2
            // we want: wnew = wold + dw1 + dw2 = wnew1 + wnew2 - wold
            // seems like we should keep a copy of the old weights, otherwise cannot compute
            #ifdef MPI_AVAILABLE
            StatefulTimer::timeCheck("allreduce START");
            WeightsPersister::copyNetWeightsToArray( net, newWeights );
            StatefulTimer::timeCheck("allreduce done copyNetWeightsToArray");
            if( myrank == 0 ) {
                for( int i = 0; i < totalWeightsSize; i++ ) {
                    weightsChange[i] = newWeights[i];
                }
            } else {
                for( int i = 0; i < totalWeightsSize; i++ ) {
                    weightsChange[i] = newWeights[i] - weightsCopy[i];
                }
            }
            MPI_Allreduce( weightsChange, weightsChangeReduced, totalWeightsSize, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD );
            StatefulTimer::timeCheck("allreduce done Allreduce");
            WeightsPersister::copyArrayToNetWeights( weightsChangeReduced, net );
            StatefulTimer::timeCheck("allreduce END");
            #endif            
        }
        StatefulTimer::dump(true);
        if( myrank == 0 ) cout << "       loss L: " << loss << endl;
        if( myrank == 0 ) timer.timeCheck("after epoch " + toString(epoch) );
//        net->print();
        if( myrank == 0 ) std::cout << "train accuracy: " << trainNumRight << "/" << trainTotalNumber << " " << (trainNumRight * 100.0f/ trainTotalNumber) << "%" << std::endl;
        if( myrank == 0 ) printAccuracy( "test", net, imagesTest, labelsTest, batchSize, config.numTest );
        if( myrank == 0 ) timer.timeCheck("after tests");
//        if( config.restartable ) {
//            WeightsPersister::persistWeights( config.restartableFilename, net );
//        }
    }
    delete[] weightsCopy;

    if( myrank == 0 ) printAccuracy( "test", net, imagesTest, labelsTest, batchSize, config.numTest );
    if( myrank == 0 ) timer.timeCheck("after tests");

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
        net->forward( &(imagesTest[batchStart][0][0]) );
        float const*outputTest = net->getOutput();
        totalNumber += thisBatchSize;
        totalNumRight += net->calcNumRight( &(labelsTest[batchStart]) );
    }
    if( myrank == 0 ) cout << "test accuracy : " << totalNumRight << "/" << totalNumber << endl;

    delete net;

    delete[] labelsTest;
//    ImagesHelper::deleteImages( &imagesTest, Ntest, imageSize );

    delete[] labels;
//    ImagesHelper::deleteImages( &imagesFloat, N, imageSize );
}

int main( int argc, char *argv[] ) {
    #ifdef MPI_AVAILABLE
        MPI_Init( &argc, &argv );
        MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
        MPI_Comm_size( MPI_COMM_WORLD, &mysize );
    #endif

    Config config;
    if( myrank == 0 ) {
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
//            cout << "    restartable=[weights are persistent?] (" << config.restartable << ")" << endl;
//            cout << "    restartablefilename=[filename to store weights] (" << config.restartableFilename << ")" << endl;
//            cout << "    outputfilename=[filename to store output] (" << config.outputFilename << ")" << endl;
        } 
    }
    for( int i = 1; i < argc; i++ ) {
       vector<string> splitkeyval = split( argv[i], "=" );
       if( splitkeyval.size() != 2 ) {
            if( myrank == 0  ){
                cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
            }
            #ifdef MPI_AVAILABLE
            MPI_Finalize();
            #endif
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
//           if( key == "restartable" ) config.restartable = atoi(value);
//           if( key == "restartablefilename" ) config.restartableFilename = value;
//           if( key == "outputfilename" ) config.outputFilename = value;
       }
    }
    go( config );
    #ifdef MPI_AVAILABLE
    MPI_Finalize();
    #endif
}


