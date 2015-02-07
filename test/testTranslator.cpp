#include <iostream>
#include <string>

#include "NorbLoader.h"
#include "BoardPng.h"
#include "Translator.h"

using namespace std;

void go( string dataDir, string setName, int n, int translateRows, int translateCols ) {
    int N;
    int numPlanes;
    int boardSize;
    string filePath = dataDir + "/" + setName + "-dat.mat";
    NorbLoader::getDimensions( filePath, &N, &numPlanes, &boardSize );
    N = n + 1;
    unsigned char *imagesUchar = new unsigned char[ N * numPlanes * boardSize * boardSize ];
    int *labels = new int[ N ];
    NorbLoader::load( filePath, imagesUchar, labels, 0, N );
    cout << "n " << n << " N " << N << endl;
    float *images = new float[ N * numPlanes * boardSize * boardSize ];
    for( int i = 0; i < N * numPlanes * boardSize * boardSize; i++ ) {
        images[i] = imagesUchar[i];
    }
    BoardPng::writeBoardsToPng( "testTranslator-1.png", images + n * numPlanes * boardSize * boardSize, numPlanes, boardSize );
    float *translated = new float[N * numPlanes * boardSize * boardSize];
    Translator::translate( n, numPlanes, boardSize, translateRows, translateCols, images, translated );
    BoardPng::writeBoardsToPng( "testTranslator-2.png", translated + n * numPlanes * boardSize * boardSize, numPlanes, boardSize );
}

int main( int argc, char *argv[] ) {
    if( argc != 6 ) {
        cout << "Usage: [datadir] [setname] [n] [translaterows] [translatecols]" << endl;
        return -1;
    }
    string dataDir = string( argv[1] );
    string setName = string( argv[2] );
    int n = atoi( argv[3] );
    int translateRows = atoi( argv[4] );
    int translateCols = atoi( argv[5] );
    go( dataDir, setName, n, translateRows, translateCols );
    return 0;
}


