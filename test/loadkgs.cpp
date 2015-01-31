#include <iostream>
using namespace std;
#include "FileHelper.h"
#include "Timer.h"
#include "stringhelper.h"
#include "NeuralNet.h"
#include "AccuracyHelper.h"
#include "BoardHelper.h"

int main(int argc, char *argv[] ) {
    const float learningRate = 0.0002f;
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
    int inputPlanes = 2;
    float *images = new float[batchSize * boardSizeSquared * inputPlanes ];
    int N = (int)(fileSize / recordSize);
    cout << "num records: " << N << endl;
    int numBatches = min( 8, N / batchSize );
    NeuralNet *net = NeuralNet::maker()->instance();
    net->inputMaker<float>()->numPlanes(inputPlanes)->boardSize(boardSize)->insert();
    for( int i = 0; i < 2; i++ ) {
        net->convolutionalMaker()->numFilters(32)->filterSize(5)->relu()->biased()->padZeros()->insert();
    }
    net->fullyConnectedMaker()->numPlanes(1)->boardSize(boardSize)->linear()->biased()->insert();
    net->softMaxLossMaker()->perPlane()->insert();
    net->setBatchSize(batchSize);
    net->print();
    int *labels = new int[ batchSize ];
    for( int epoch = 0; epoch < 200; epoch++ ) {
        int nBatch = 0;
        while( nBatch < numBatches ) {
            int count = 0;
            unsigned char *kgsData = reinterpret_cast<unsigned char *>( FileHelper::readBinaryChunk( dataFilePath, (long)nBatch * batchSize * recordSize, (long)batchSize * recordSize ) );
    //        timer.timeCheck("read chunk from file");
            memset( images, 0, sizeof(float) * batchSize * inputPlanes * boardSizeSquared );
            for( int n = 0; n < batchSize; n++ ) {
                int intraChunkPos = n * recordSize;
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
                labels[ n ] = label;
//                cout << moveRow << "," << moveCol << " label " << label << endl;
    //            int label = moveRow * 19 + moveCol;
//                for( int inputPlane = 0; inputPlane < inputPlanes; inputPlane++ ) {
                    int board0Offset = ( n * inputPlanes + 0 ) * boardSizeSquared;
                    int board1Offset = ( n * inputPlanes + 1 ) * boardSizeSquared;
                    for( int i = 0; i < boardSizeSquared; i++ ) {
//                        images[ boardOffset + i ] = ( kgsData[intraChunkPos + 4 + i] >> inputPlane ) & 1;
                        images[ board0Offset + i ] = ( kgsData[intraChunkPos + 4 + i] & 7 ) > 0 ? 1 : 0;
                        images[ board1Offset + i ] = ( kgsData[intraChunkPos + 4 + i] & 56 ) > 0 ? 1 : 0;
                    }
//                }
//                cout << "n: " << n << endl;
//                BoardHelper::printBoard( &(images[ board0Offset]), boardSize );
//                BoardHelper::printBoard( &(images[ board1Offset]), boardSize );
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
            }
            int numRight = net->calcNumRight( labels );
            // now train on it
            net->learnBatchFromLabels( learningRate, images, labels );
            float loss = net->calcLossFromLabels( labels );
            cout << "loss: " << loss << " test accuracy " << numRight << "/" << batchSize << " " << ( numRight * 100.0f / batchSize ) << "%" << " batch " << nBatch << endl;
            if( nBatch > 0 && nBatch % 70 == 0 ) cout << nBatch << endl;
            nBatch++;
        }
    }
    return 0;
}

