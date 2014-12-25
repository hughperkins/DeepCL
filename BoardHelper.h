#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

class BoardHelper {
public:
    static int **allocateBoard( int boardSize ) {
        int *contiguousarray = new int[ boardSize * boardSize ];
        memset(contiguousarray, 0, sizeof(int) * boardSize * boardSize );
        int **board = new int*[boardSize];
        for( int i = 0; i < boardSize; i++ ) {
            board[i] = &(contiguousarray[i*boardSize]);
        }
        return board;
    }

    static float **allocateBoardFloats( int boardSize ) {
        float *contiguousarray = new float[ boardSize * boardSize ];
        memset(contiguousarray, 0, sizeof(float) * boardSize * boardSize );
        float **board = new float*[boardSize];
        for( int i = 0; i < boardSize; i++ ) {
            board[i] = &(contiguousarray[i*boardSize]);
        }
        return board;
    }

    static void deleteBoard( int ***p_board, int boardSize ) {
       if( p_board == 0 ) {
          return;
       }
       delete[] (*p_board)[0];
       delete[] *p_board;
       *p_board = 0;
    }

    static void deleteBoard( float ***p_board, int boardSize ) {
       if( p_board == 0 ) {
          return;
       }
       delete[] (*p_board)[0];
       delete[] *p_board;
       *p_board = 0;
    }

    static void copyBoard( int *const *const dst, int const*const*const src, int boardSize ) {
        for( int i = 0; i < boardSize; i++ ) {
            for( int j = 0; j < boardSize; j++ ) {
                if( dst[i][j] != src[i][j] ) {
                    dst[i][j] = src[i][j];
                }
            }
        }
    }

    static void wipeBoard( int *const*const board, int boardSize ) {
        for( int i = 0; i < boardSize; i++ ) {
            for( int j = 0; j < boardSize; j++ ) {
                board[i][j] = 0;
            }
        }
    }

    static int **loadBoard( std::string filepath, int *p_boardSize ) {
        std::ifstream f;
        f.open( filepath.c_str() );
       std::string thisline;
       f >> thisline;
       *p_boardSize = thisline.length();
       if( *p_boardSize == 0 ) {
          std::cout << "boardhelper::loadBoard. error: boardsize 0, " << filepath << std::endl;
          throw "boardhelper::loadBoard. error: boardsize 0 " + filepath;
       }
       int **board = allocateBoard( *p_boardSize );
        for( int i = 0; i < *p_boardSize; i++ ) {
           if( i == 0 ) {
           }
           for( int j = 0; j < *p_boardSize; j++ ) {
              std::string thischar = std::string("") + thisline[j];
              if( thischar == "*" ) {
                  (board)[i][j] = 1;
              }
              if( thischar == "O" ) {
                  (board)[i][j] = 2;
              }
           }
           f >> thisline;
        }
        f.close();
        return board;
    }
};


