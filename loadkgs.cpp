#include <iostream>
using namespace std;
#include "FileHelper.h"
#include "Timer.h"
#include "stringhelper.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"

int main(int argc, char *argv[] ) {
    const float learningRate = 0.01f;
    const int batchSize = 128;

    Timer timer;
    std::string dataFilePath = "../data/kgsgo/kgsgo.dat";
    long fileSize = FileHelper::getFilesize( dataFilePath );
    timer.timeCheck("read file, size " + toString(fileSize/1024/1024) + "MB");
//    int i = 0;
    long pos = 0;
    const int boardSize = 19;
    const int boardSizeSquared = boardSize * boardSize;
    const int recordSize = 2 + 2 + 19 * 19;
    cout << "recordsize: " << recordSize << endl;
    int inputPlanes = 8;
    float *images = new float[batchSize * boardSizeSquared * inputPlanes ];
    int N = (int)(fileSize / recordSize);
    cout << "num records: " << N << endl;
    int numBatches = N / batchSize;
    int nBatch = 0;
    NeuralNet *net = NeuralNet::maker()->planes(8)->boardSize(boardSize)->instance();
    for( int i = 0; i < 1; i++ ) {
        net->convolutionalMaker()->numFilters(128)->filterSize(3)->relu()->biased()->padZeros()->insert();
    }
    net->convolutionalMaker()->numFilters(19*19)->filterSize(net->layers[net->layers.size()-1]->boardSize)->tanh()->biased()->insert();
    net->setBatchSize(batchSize);
    int *labels = new int[ boardSizeSquared ];
    float *expectedValues = new float[ batchSize * boardSizeSquared ];
//    while( pos < fileSize ) {
    while( nBatch < numBatches ) {
        int count = 0;
        for( int i = 0; i < batchSize * boardSizeSquared; i++ ) {
            expectedValues[i] = -0.5f;
        }
        char *kgsData = FileHelper::readBinaryChunk( dataFilePath, (long)nBatch * batchSize * recordSize * sizeof(float),
            (long)batchSize * recordSize * sizeof(float) );
        int intraChunkPos = 0;
        while( count < batchSize ) {
            //cout << kgsData[pos];
            //if( count % 80 == 0 && count != 0 ) cout << endl;
            int moveRow = kgsData[pos + 2];
            int moveCol = kgsData[pos + 3];
            int label = moveRow * 19 + moveCol;
            labels[ count ] = label;
            expectedValues[ count * boardSizeSquared + label ] = 0.5f;
            //cout << moveRow << "," << moveCol << endl;
//            int label = moveRow * 19 + moveCol;
            if( kgsData[intraChunkPos] != 'G' ) {
                throw std::runtime_error("alignment error, for intrachunkpos " + toString(intraChunkPos) );
            }
            for( int inputPlane = 0; inputPlane < inputPlanes; inputPlane++ ) {
                int boardOffset = ( count * inputPlanes + inputPlane ) * boardSizeSquared;
                for( int i = 0; i < boardSizeSquared; i++ ) {
                    images[ boardOffset + i ] = ( kgsData[intraChunkPos + 4 + i] >> inputPlane ) & 1;
                }
            }
            intraChunkPos += recordSize;
            count++;
        }
        // use this batch first as a test set, to test accuracy
        net->propagate( images );
        float const*resultsTest = net->getResults();
        int numRight = AccuracyHelper::calcNumRight( batchSize, 19 * 19, labels, resultsTest );
        cout << "test accuracy " << numRight << "/" << batchSize << " " << ( numRight * 100.0f / batchSize ) << "%" << endl;
        // now train on it
        net->learnBatch( learningRate, images, expectedValues );
        float loss = net->calcLoss( expectedValues );
        cout << "loss: " << loss << endl;
        cout << ".";
        if( nBatch > 0 && nBatch % 70 == 0 ) cout << nBatch << endl;
        nBatch++;
//        timer.timeCheck("processed batch of " + toString( batchSize ) );
    }
    return 0;
}

