#include <iostream>

#include "NorbLoader.h"
#include "BoardPng.h"

using namespace std;

int main( int argc, char *argv[] ) {
    if( argc != 4 ) {
        cout << "Usage: " << argv[0] << " [datadir] [setname] [num images]" << endl;
        return -1;
    }
    string dataDir = argv[1];
    string setName = argv[2];
//    int startIndex = atoi(argv[3]);
    int num = atoi(argv[3]);
    int N;
    int numPlanes;
    int boardSize;
    unsigned char *images = NorbLoader::loadImages( dataDir + "/" + setName + "-dat.mat", &N, &numPlanes, &boardSize, num );
    BoardPng::writeBoardsToPng( "printMat-output.png", images, num, boardSize );
    return 0;
}

