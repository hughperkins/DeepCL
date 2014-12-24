// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "BoardHelper.h"
#include "MyException.h"

class BoardsHelper {
public:
    static int ***allocateBoards( int N, int boardSize ) {
        int *contiguousSpace = new int[N * boardSize * boardSize ];
        if( contiguousSpace == 0 ) {
            throw MyException("failed to allocate memory");
        }
        int ***boards = new int**[N];
        if( boards == 0 ) {
            throw MyException("failed to allocate int***boards memory");
        }
        for( int n = 0; n < N; n++ ) {
           int **board = new int*[boardSize];
           int *thisboardcontiguousspace = &(contiguousSpace[n * boardSize * boardSize]);
            if( board == 0 ) {
                throw MyException("failed to allocate int **board memory");
            }
           boards[n] = board;
           for( int i = 0; i < boardSize; i++ ) {
              board[i] = &(thisboardcontiguousspace[i * boardSize ]);
           }
        }

//        int ***boards = new int**[N];
//        for( int i = 0; i < N; i++ ) {
//            boards[i] = BoardHelper::allocateBoard( boardSize );
//        }
        return boards;
    }

    static void deleteBoards( int ****p_boards, int N, int boardSize ) {
        int ***boards = *p_boards;
        int *contiguous = &(boards[0][0][0] );
        for( int n = 0; n < N; n++ ) {
            delete[] boards[n];
        }
        delete[] boards;       
        delete[] contiguous;

//        for( int n = 0; n < N; n++ ) {
//            BoardHelper::deleteBoard( &(*boards)[n], boardSize );
//        }
//        delete[] (*boards);
        *p_boards = 0;
    }

    static float ***allocateBoardsFloat( int N, int boardSize ) {
        float *contiguousSpace = new float[N * boardSize * boardSize ];
        if( contiguousSpace == 0 ) {
            throw MyException("failed to allocate memory");
        }
        float ***boards = new float**[N];
        if( boards == 0 ) {
            throw MyException("failed to allocate int***boards memory");
        }
        for( int n = 0; n < N; n++ ) {
           float **board = new float*[boardSize];
           float *thisboardcontiguousspace = &(contiguousSpace[n * boardSize * boardSize]);
            if( board == 0 ) {
                throw MyException("failed to allocate int **board memory");
            }
           boards[n] = board;
           for( int i = 0; i < boardSize; i++ ) {
              board[i] = &(thisboardcontiguousspace[i * boardSize ]);
           }
        }

//        int ***boards = new int**[N];
//        for( int i = 0; i < N; i++ ) {
//            boards[i] = BoardHelper::allocateBoard( boardSize );
//        }
        return boards;
    }

    static void deleteBoardsFloat( float ****p_boards, int N, int boardSize ) {
        float ***boards = *p_boards;
        float *contiguous = &(boards[0][0][0] );
        for( int n = 0; n < N; n++ ) {
            delete[] boards[n];
        }
        delete[] boards;       
        delete[] contiguous;

//        for( int n = 0; n < N; n++ ) {
//            BoardHelper::deleteBoard( &(*boards)[n], boardSize );
//        }
//        delete[] (*boards);
        *p_boards = 0;
    }

    static void copyBoards( float ***target, int ***source, int N, int boardSize ) {
        float *target1d = &(target[0][0][0]);
        int *source1d = &(source[0][0][0]);
        const int T = N * boardSize * boardSize;
        for( int i = 0; i < T; i++ ) {
            target1d[i] = source1d[i];
        }
    }
};

