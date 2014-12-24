#include <iostream>

#include "BoardsHelper.h"
#include "MnistLoader.h"
#include "BoardPng.h"

using namespace std;

int main( int argc, char *argv[] ) {
//    MemoryChecker memoryChecker;
    int ***boards = BoardsHelper::allocateBoards( 5, 11 );
    BoardsHelper::deleteBoards( &boards, 5, 11 );

//    int N = 1000;
    int N;
    int boardSize;
    boards = MnistLoader::loadImages( "/norep/Downloads/data/mnist", "train", &N, &boardSize );
    int ***results = BoardsHelper::allocateBoards( N, boardSize );

    int *boards_1d = &(boards[0][0][0]);
    for( int i = 0; i < N * boardSize * boardSize; i++ ) {
        boards_1d[i] = 0x55;
    }

    BoardPng::writeBoardsToPng( "foo.png", boards, min(100, N ), boardSize );
    BoardPng::writeBoardsToPng( "foo.png", boards, min(100, N ), boardSize );
    BoardPng::writeBoardsToPng( "foo.png", boards, min(100, N ), boardSize );
    BoardPng::writeBoardsToPng( "foo.png", boards, min(100, N ), boardSize );

    BoardsHelper::deleteBoards( &results, N, boardSize );
    BoardsHelper::deleteBoards( &boards, N, boardSize );
    return 0;
}

