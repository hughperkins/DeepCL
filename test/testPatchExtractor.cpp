#include <iostream>
#include <string>

#include "NorbLoader.h"
#include "BoardPng.h"
#include "PatchExtractor.h"

using namespace std;

void go( string dataDir, string setName, int n, int patchSize, int patchRow, int patchCol ) {
    int N;
    int numPlanes;
    int boardSize;
    unsigned char *imagesUchar = NorbLoader::loadImages( dataDir + "/" + setName + "-dat.mat", &N, &numPlanes, &boardSize, n + 1 );
    cout << "n " << n << " N " << N << endl;
    float *images = new float[ N * numPlanes * boardSize * boardSize ];
    for( int i = 0; i < N * numPlanes * boardSize * boardSize; i++ ) {
        images[i] = imagesUchar[i];
    }
    BoardPng::writeBoardsToPng( "testPatchExtractor-1.png", images + n * numPlanes * boardSize * boardSize, numPlanes, boardSize );
    float *patches = new float[N * numPlanes * patchSize * patchSize];
    PatchExtractor::extractPatch( n, numPlanes, boardSize, patchSize, patchRow, patchCol, images, patches );
    BoardPng::writeBoardsToPng( "testPatchExtractor-2.png", patches + n * numPlanes * patchSize * patchSize, numPlanes, patchSize );
}

int main( int argc, char *argv[] ) {
    if( argc != 7 ) {
        cout << "Usage: [datadir] [setname] [n] [patchsize] [patchrow] [patchcol]" << endl;
        return -1;
    }
    string dataDir = string( argv[1] );
    string setName = string( argv[2] );
    int n = atoi( argv[3] );
    int patchSize = atoi( argv[4] );
    int patchRow = atoi( argv[5] );
    int patchCol = atoi( argv[6] );
    go( dataDir, setName, n, patchSize, patchRow, patchCol );
    return 0;
}


