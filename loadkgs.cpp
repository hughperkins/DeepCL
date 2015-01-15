#include <iostream>
using namespace std;
#include "FileHelper.h"
#include "Timer.h"
#include "stringhelper.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"

int main(int argc, char *argv[] ) {
    const float learningRate = 0.0001f;
    const int batchSize = 128;
    const int labelGrouping = 1;

    const int numLabelRows = ((19+labelGrouping-1)/labelGrouping);
    const int numLabels = numLabelRows * numLabelRows;
    cout << " labelgrouping " << labelGrouping << " numLabelRows " << numLabelRows << " numLabels " << numLabels << endl;
//return -1;
    Timer timer;
    std::string dataFilePath = "/home/user/git/kgsgo-dataset-preprocessor/data/KGS-2011_06-19-1419-.dat";
//    std::string dataFilePath = "/home/user/git/kgsgo-dataset-preprocessor/data/kgsgo.dat";
    long fileSize = FileHelper::getFilesize( dataFilePath );
    timer.timeCheck("read file, size " + toString(fileSize/1024/1024) + "MB");
//    int i = 0;
    long pos = 0;
    const int boardSize = 19;
    const int boardSizeSquared = boardSize * boardSize;
    const int recordSize = 2 + 2 + boardSizeSquared;
    cout << "recordsize: " << recordSize << endl;
    int inputPlanes = 8;
    float *images = new float[batchSize * boardSizeSquared * inputPlanes ];
    int N = (int)(fileSize / recordSize);
    cout << "num records: " << N << endl;
    int numBatches = N / batchSize;
    int nBatch = 0;
    NeuralNet *net = NeuralNet::maker()->planes(8)->boardSize(boardSize)->instance();
    for( int i = 0; i < 1; i++ ) {
        net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->padZeros()->insert();
    }
    net->convolutionalMaker()->numFilters(numLabels)->filterSize(net->layers[net->layers.size()-1]->boardSize)->tanh()->biased()->insert();
    net->setBatchSize(batchSize);
    int *labels = new int[ batchSize ];
    float *expectedValues = new float[ batchSize * numLabels ];
//    while( pos < fileSize ) {
    while( nBatch < numBatches ) {
        int count = 0;
        for( int i = 0; i < batchSize * numLabels; i++ ) {
            expectedValues[i] = -0.5f;
        }
        char *kgsData = FileHelper::readBinaryChunk( dataFilePath, (long)nBatch * batchSize * recordSize, (long)batchSize * recordSize );
//        timer.timeCheck("read chunk from file");
        int intraChunkPos = 0;
        memset( images, 0, sizeof(float) * boardSizeSquared * inputPlanes * batchSize );
        while( count < batchSize ) {
            //cout << kgsData[pos];
            //if( count % 80 == 0 && count != 0 ) cout << endl;
            if( kgsData[intraChunkPos] != 'G' ) {
                throw std::runtime_error("alignment error, for intrachunkpos " + toString(intraChunkPos) );
            }
            int moveRow = kgsData[intraChunkPos + 2];
            int moveCol = kgsData[intraChunkPos + 3];
//            cout << moveRow << "," << moveCol << endl;
            moveRow = moveRow / labelGrouping;
            moveCol = moveCol / labelGrouping;
            int label = moveRow * numLabelRows + moveCol;
            labels[ count ] = label;
            expectedValues[ count * numLabels + label ] = 0.5f;
//            cout << moveRow << "," << moveCol << " label " << label << endl;
//            int label = moveRow * 19 + moveCol;
            for( int inputPlane = 0; inputPlane < inputPlanes; inputPlane++ ) {
                int boardOffset = ( count * inputPlanes + inputPlane ) * boardSizeSquared;
                for( int i = 0; i < boardSizeSquared; i++ ) {
                    images[ boardOffset + i ] = ( kgsData[intraChunkPos + 4 + i] >> inputPlane ) & 1;
                }
            }
            intraChunkPos += recordSize;
            count++;
        }
//        timer.timeCheck("copy to array");
        // use this batch first as a test set, to test accuracy
        net->propagate( images );
//        timer.timeCheck("propagate, for test");
        float const*resultsTest = net->getResults();
        for( int n = 0; n < batchSize; n++ ) {
//            cout << "n " << n << endl;
//            cout << "label: " << labels[n] << endl;
//            for( int i = 0; i < numLabelRows; i++ ) {
//                for( int j = 0; j < numLabelRows; j++ ) {
//                    if( labels[n] == ( i * numLabelRows + j ) ) {
//                        cout << "*";
//                    } else {
//                        cout << ".";
//                    }
//                }
//                cout << endl;
//            }
//            cout << endl;
            int thisboard[boardSizeSquared];
            memset( thisboard, 0, sizeof(int) * boardSizeSquared );
            for( int inputPlane = 0; inputPlane < 6; inputPlane++ ) {
                for( int i = 0; i < boardSize; i++ ) {
                    for( int j = 0; j < boardSize; j++ ) {
                        if( images[ n * boardSizeSquared * inputPlanes + inputPlane * boardSizeSquared + i * boardSize + j ] == 1 ) {
                            if( inputPlane < 3 ) {
                                thisboard[i*boardSize+j] = 1;
                            } else if( inputPlane < 6 ) {
                                thisboard[i*boardSize+j] = 2;
                            }
                        }
                    }
                }
            }
//            for( int i = 0; i < boardSize; i++ ) {
//                for( int j = 0; j < boardSize; j++ ) {
//                    if( thisboard[i*boardSize+j] == 1 ) {
//                        cout << "*";
//                    } else if( thisboard[i*boardSize+j] == 2 ) {
//                        cout << "O";
//                    } else {
//                        cout << ".";
//                    } 
//                }
//                cout << endl;
//            }
//            cout << endl;
        }
        int numRight = AccuracyHelper::calcNumRight( batchSize, numLabels, labels, resultsTest );
        // now train on it
        net->learnBatch( learningRate, images, expectedValues );
//        timer.timeCheck("learn one batch");
        float loss = net->calcLoss( expectedValues );
        cout << "loss: " << loss << " test accuracy " << numRight << "/" << batchSize << " " << ( numRight * 100.0f / batchSize ) << "%" << " batch " << nBatch << endl;
//        cout << ".";
        if( nBatch > 0 && nBatch % 70 == 0 ) cout << nBatch << endl;
        nBatch++;
//        timer.timeCheck("processed batch of " + toString( batchSize ) );
    }
    return 0;
}

