// Copyright Hugh Perkins 2013 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once
#include <cmath>

#include "png++/png.hpp"

class BoardPng {
public:
    static int getBoardMax( int ** board, int boardSize ) {
        int maxvalue = 0;
        for( int i = 0; i < boardSize; i++ ) {
            for( int j = 0; j < boardSize; j++ ) {
                maxvalue = std::max( maxvalue, board[i][j] );
            }
        }
        return maxvalue;
    }

    static float getBoardMax( float ** board, int boardSize ) {
        float maxvalue = 0;
        for( int i = 0; i < boardSize; i++ ) {
            for( int j = 0; j < boardSize; j++ ) {
                maxvalue = std::max( maxvalue, board[i][j] );
            }
        }
        return maxvalue;
    }

    static float getBoardMax( float const* board, int boardSize ) {
        float maxvalue = 0;
        for( int i = 0; i < boardSize; i++ ) {
            for( int j = 0; j < boardSize; j++ ) {
                maxvalue = std::max( maxvalue, board[i*boardSize + j] );
            }
        }
        return maxvalue;
    }
    static float getBoardMin( float const* board, int boardSize ) {
        float minvalue = 0;
        for( int i = 0; i < boardSize; i++ ) {
            for( int j = 0; j < boardSize; j++ ) {
                minvalue = std::min( minvalue, board[i*boardSize + j] );
            }
        }
        return minvalue;
    }
//    static float getBoardMax( unsigned char const* board, int boardSize ) {
//        float maxvalue = 0;
//        for( int i = 0; i < boardSize; i++ ) {
//            for( int j = 0; j < boardSize; j++ ) {
//                maxvalue = std::max( maxvalue, board[i*boardSize + j] );
//            }
//        }
//        return maxvalue;
//    }
//    static float getBoardMin( unsigned char const* board, int boardSize ) {
//        float minvalue = 0;
//        for( int i = 0; i < boardSize; i++ ) {
//            for( int j = 0; j < boardSize; j++ ) {
//                minvalue = std::min( minvalue, board[i*boardSize + j] );
//            }
//        }
//        return minvalue;
//    }

    static void writeBoardToPng( std::string filename, int **board, int boardSize ) {
        int maxvalue = getBoardMax( board, boardSize );
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >( boardSize, boardSize );
        for( int i = 0; i < boardSize; i++ ) {
            for( int j = 0; j < boardSize; j++ ) {
               (*image)[i][j] = png::rgb_pixel( board[i][j] * 255 / maxvalue, board[i][j] * 255 / maxvalue, board[i][j] * 255 / maxvalue );
            }
        }
        remove( filename.c_str() );
        image->write( filename );
        delete image;
    }

    static void writeBoardsToPng( std::string filename, int ***boards, int numBoards, int boardSize ) {
        int cols = sqrt( numBoards );
        if( cols * cols < numBoards ) {
            cols++;
        }
        int rows = ( numBoards + cols - 1 ) / cols;
        std::cout << "numBoards " << numBoards << " rows " << rows << " cols " << cols << std::endl;
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >( boardSize * rows, boardSize * cols );


        for( int x = 0; x < cols; x++ ) {
           for( int y = 0; y < rows; y++ ) {
                if( x * rows + y >= numBoards ) {
                    continue;
                }
//                cout << "board at x " << x << " y " << y << endl;
                int **board = boards[x*rows + y];
                int maxvalue = std::max( 1, getBoardMax( board, boardSize ) );
                for( int i = 0; i < boardSize; i++ ) {
                    for( int j = 0; j < boardSize; j++ ) {
                       (*image)[x*boardSize + i][y*boardSize + j] = png::rgb_pixel( board[i][j] * 255 / maxvalue, board[i][j] * 255 / maxvalue, board[i][j] * 255 / maxvalue );
                    }
                }

            }
         }
        remove( filename.c_str() );
        image->write( filename );
        delete image;
    }

    static void writeBoardsToPng( std::string filename, float ***boards, int numBoards, int boardSize ) {
        int cols = sqrt( numBoards );
        if( cols * cols < numBoards ) {
            cols++;
        }
        int rows = ( numBoards + cols - 1 ) / cols;
        std::cout << "numBoards " << numBoards << " rows " << rows << " cols " << cols << std::endl;
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >( boardSize * rows, boardSize * cols );


        for( int x = 0; x < cols; x++ ) {
           for( int y = 0; y < rows; y++ ) {
                if( x * rows + y >= numBoards ) {
                    continue;
                }
//                cout << "board at x " << x << " y " << y << endl;
                float **board = boards[x*rows + y];
                float maxvalue = std::max( 1.0f, getBoardMax( board, boardSize ) );
                for( int i = 0; i < boardSize; i++ ) {
                    for( int j = 0; j < boardSize; j++ ) {
                       (*image)[x*boardSize + i][y*boardSize + j] = png::rgb_pixel( board[i][j] * 255 / maxvalue, board[i][j] * 255 / maxvalue, board[i][j] * 255 / maxvalue );
                    }
                }

            }
         }
        remove( filename.c_str() );
        image->write( filename );
        delete image;
    }

    static void writeBoardsToPng( std::string filename, float const*boards, int numBoards, int boardSize ) {
        int cols = sqrt( numBoards );
        if( cols * cols < numBoards ) {
            cols++;
        }
        int rows = ( numBoards + cols - 1 ) / cols;
        std::cout << "numBoards " << numBoards << " rows " << rows << " cols " << cols << std::endl;
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >( boardSize * rows, boardSize * cols );


        for( int x = 0; x < cols; x++ ) {
           for( int y = 0; y < rows; y++ ) {
                if( x * rows + y >= numBoards ) {
                    continue;
                }
//                cout << "board at x " << x << " y " << y << endl;
                float const*board = &(boards[boardSize * boardSize * ( x*rows + y ) ]);
                float maxValue = getBoardMax( board, boardSize );
                float minValue = getBoardMin( board, boardSize );
                for( int i = 0; i < boardSize; i++ ) {
                    for( int j = 0; j < boardSize; j++ ) {
                       float normValue = ( board[i*boardSize + j] + minValue ) * 255.0f / (maxValue - minValue );
                       (*image)[x*boardSize + i][y*boardSize + j] = png::rgb_pixel( normValue, normValue, normValue );
                    }
                }

            }
         }
        remove( filename.c_str() );
        image->write( filename );
        delete image;
    }
    static void writeBoardsToPng( std::string filename, unsigned char const*boards, int numBoards, int boardSize ) {
        int cols = sqrt( numBoards );
        if( cols * cols < numBoards ) {
            cols++;
        }
        int rows = ( numBoards + cols - 1 ) / cols;
        std::cout << "numBoards " << numBoards << " rows " << rows << " cols " << cols << std::endl;
        png::image< png::rgb_pixel > *image = new png::image< png::rgb_pixel >( boardSize * rows, boardSize * cols );


        for( int x = 0; x < cols; x++ ) {
           for( int y = 0; y < rows; y++ ) {
                if( x * rows + y >= numBoards ) {
                    continue;
                }
//                cout << "board at x " << x << " y " << y << endl;
                unsigned char const*board = &(boards[boardSize * boardSize * ( x*rows + y ) ]);
                float maxValue = 255; // getBoardMax( board, boardSize );
                float minValue = 0; // getBoardMin( board, boardSize );
                for( int i = 0; i < boardSize; i++ ) {
                    for( int j = 0; j < boardSize; j++ ) {
                       float normValue = ( board[i*boardSize + j] + minValue ) * 255.0f / (maxValue - minValue );
                       (*image)[x*boardSize + i][y*boardSize + j] = png::rgb_pixel( normValue, normValue, normValue );
                    }
                }

            }
         }
        remove( filename.c_str() );
        image->write( filename );
        delete image;
    }
};

