#include <iostream>
#include <string>

//#include "NorbLoader.h"
#include "BoardPng.h"
#include "GenericLoader.h"
#include "NormalizationHelper.h"

using namespace std;

void go( string trainFilepath, int startN, int numExamples ) {
    int N;
    int numPlanes;
    int boardSize;
    int totalSize;
    GenericLoader::getDimensions( trainFilepath, &N, &numPlanes, &boardSize, &totalSize );
    cout << "N " << N << " numplanes " << numPlanes << " boardSize " << boardSize << endl;
    unsigned char *images = new unsigned char[ numExamples * numPlanes * boardSize * boardSize ];
    int *labels = new int[ numExamples ];
    GenericLoader::load( trainFilepath, images, labels, startN, numExamples );
//    float *images = new float[ N * numPlanes * boardSize * boardSize ];
//    for( int i = 0; i < N * numPlanes * boardSize * boardSize; i++ ) {
//        images[i] = imagesUchar[i];
//    }
    float thismin;
    float thismax;
    NormalizationHelper::getMinMax( images, numExamples * numPlanes * boardSize * boardSize, &thismin, &thismax );
    cout << "min: " << thismin << " max: " << thismax << endl;
    BoardPng::writeBoardsToPng( "testGenericLoader.png", images, numExamples * numPlanes, boardSize );
    for( int i = 0; i < numExamples; i++ ) {
        cout << "labels[" << i << "]=" << labels[i] << endl;
    }
//    float *translated = new float[N * numPlanes * boardSize * boardSize];
//    Translator::translate( n, numPlanes, boardSize, translateRows, translateCols, images, translated );
//    BoardPng::writeBoardsToPng( "testTranslator-2.png", translated + n * numPlanes * boardSize * boardSize, numPlanes, boardSize );
}

int main( int argc, char *argv[] ) {
    if( argc != 4 ) {
        cout << "Usage: [trainfilepath] [startn] [numexamples]" << endl;
        return -1;
    }
    string trainFilepath = string( argv[1] );
    int startN = atoi( argv[2] );
    int numExamples = atoi( argv[3] );
    go( trainFilepath, startN, numExamples );
    return 0;
}


