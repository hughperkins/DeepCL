#include <iostream>

#include "MnistLoader.h"
#include "BoardPng.h"

using namespace std;

int main( int argc, char *argv[] ) {
    string mnistDir = argv[1];
    string setName = argv[2];
    int startIndex = atoi(argv[3]);
    int num = atoi(argv[4]);
    int N;
    int boardSize;
    int ***boards = MnistLoader::loadImages( mnistDir, setName, &N, &boardSize );
    BoardPng::writeBoardsToPng( "printMnist-output.png", &(boards[startIndex]), num, boardSize );
    return 0;
}

