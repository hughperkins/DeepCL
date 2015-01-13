#include <iostream>
using namespace std;
#include "FileHelper.h"
#include "Timer.h"
#include "stringhelper.h"
#include "NeuralNet.h"

int main(int argc, char *argv[] ) {
    int size;
    Timer timer;
    char *kgsData = FileHelper::readBinary( "../../kgsgo-dataset-preprocessor/data/kgsgo.dat", &size );
    timer.timeCheck("read file, size " + toString(size/1024/1024) + "MB");
//    int i = 0;
    int pos = 0;
    const int batchSize = 128;
    const int boardSize = 19;
    const int boardSizeSquared = boardSize * boardSize;
    const int recordSize = 2 + 2 + 19 * 19;
    cout << "recordsize: " << recordSize << endl;
    int inputPlanes = 8;
    float *images = new float[batchSize * boardSizeSquared * inputPlanes ];
    int N = size / recordSize;
    cout << "num records: " << N << endl;
    int nBatch = 0;
    NeuralNet *net = NeuralNet::maker()->planes(8)->boardSize(boardSize)->instance();
    for( int i = 0; i < 1; i++ ) {
        net->convolutionalMaker()->numFilters(128)->filterSize(3)->relu()->biased()->padZeros()->insert();
    }
    net->convolutionalMaker()->numFilters(19*19)->filterSize(net->layers[net->layers.size()-1]->boardSize)->tanh()->biased()->insert();
    net->
    net->setBatchSize(batchSize);
    while( pos < size ) {
        int count = 0;
        while( count < batchSize ) {
            //cout << kgsData[pos];
            //if( count % 80 == 0 && count != 0 ) cout << endl;
            pos += 19*19 + 2 + 2;
            int moveRow = kgsData[pos + 2];
            int moveCol = kgsData[pos + 3];
            //cout << moveRow << "," << moveCol << endl;
            int label = moveRow * 19 + moveCol;
            for( int inputPlane = 0; inputPlane < inputPlanes; inputPlane++ ) {
                int boardOffset = ( count * inputPlanes + inputPlane ) * boardSizeSquared;
                for( int i = 0; i < boardSizeSquared; i++ ) {
                    images[ boardOffset + i ] = ( kgsData[pos + 4 + i] >> inputPlane ) & 1;
                }
            }
            count++;
        }
        cout << ".";
        if( nBatch > 0 && nBatch % 70 == 0 ) cout << nBatch << endl;
        nBatch++;
//        timer.timeCheck("processed batch of " + toString( batchSize ) );
    }
    return 0;
}

