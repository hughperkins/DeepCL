// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>

#include "FileHelper.h"
#include "BoardsHelper.h"

class MnistLoader {
public:
    static int **loadImage( string dir, string set, int idx, int *p_size ) {
        int imagesFilesize = 0;
        int labelsFilesize = 0;
        unsigned char *imagesData = FileHelper::readBinary( dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize );
        unsigned char *labelsData = FileHelper::readBinary( dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize );

        int numImages = readUInt( imagesData, 1 );
        int numRows = readUInt( imagesData, 2 );
        int numCols = readUInt( imagesData, 3 );
        *p_size = numRows;
        cout << "numimages " << numImages << " " << numRows << "*" << numCols << endl;

        int **board = BoardHelper::allocateBoard( numRows );
        for( int i = 0; i < numRows; i++ ) {
            for( int j = 0; j < numRows; j++ ) {
                board[i][j] = (int)imagesData[idx * numRows * numCols + i * numCols + j];
            }
        }
        delete[] imagesData;
        delete[] labelsData;
        return board;
    }
    static int ***loadImages( string dir, string set, int *p_numImages, int *p_size ) {
        int imagesFilesize = 0;
        unsigned char *imagesData = FileHelper::readBinary( dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize );

        int totalNumImages = readUInt( imagesData, 1 );
        int numRows = readUInt( imagesData, 2 );
        int numCols = readUInt( imagesData, 3 );
//        *p_numImages = min(100,totalNumImages);
        *p_numImages = totalNumImages;
        *p_size = numRows;
        cout << "totalNumImages " << *p_numImages << " " << *p_size << "*" << numCols << endl;
        int ***boards = BoardsHelper::allocateBoards( *p_numImages, numRows );
        for( int n = 0; n < *p_numImages; n++ ) {
            for( int i = 0; i < numRows; i++ ) {
                for( int j = 0; j < numRows; j++ ) {
                    boards[n][i][j] = (int)imagesData[16 + n * numRows * numCols + i * numCols + j];
                }
            }
        }
        delete[] imagesData;
        return boards;
    }
    static int *loadLabels( string dir, string set, int *p_numImages ) {
        int labelsFilesize = 0;
        unsigned char *labelsData = FileHelper::readBinary( dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize );

        int totalNumImages = readUInt( labelsData, 1 );
      //  *p_numImages = min(100,totalNumImages);
        *p_numImages = totalNumImages;
        cout << "set " << set << " num labels " << *p_numImages << endl;
        int *labels = new int[*p_numImages];
        for( int n = 0; n < *p_numImages; n++ ) {
           labels[n] = (int)labelsData[8 + n];
        }
        delete[] labelsData;
        return labels;
    }

    static void swap( int **tempBoard, int ***images, int *labels, int boardSize, int one, int two ) {
        int templabel = labels[one];
        labels[one] = labels[two];
        labels[two] = templabel;
        BoardHelper::copyBoard( tempBoard, images[one], boardSize );
        BoardHelper::copyBoard( images[one], images[two], boardSize );
        BoardHelper::copyBoard( images[two], tempBoard, boardSize );        
    }

    static void shuffle( int ***images, int *labels, int N, int boardSize ) {
        std::mt19937 random;
//        int boardSizeSquared = boardSize * boardSize;
        // eg N = 2
        // i = 0   random() % 2  0, 1 + i => 0, 1
        // i = 1   random() % 1  0    +i => 1
//        return;
        int **tempBoard = BoardHelper::allocateBoard( boardSize );
//        for( int i = 0; i < N - 1; i++ ) {
//            int swappos = i + ( random() % ( N - i ) );
//            if( i != swappos ) {
////                std::cout << "swap " << i << " " << swappos << std::endl;
//                int templabel = labels[i];
//                labels[i] = labels[swappos];
//                labels[swappos] = templabel;
//                BoardHelper::copyBoard( tempBoard, images[i], boardSize );
//                BoardHelper::copyBoard( images[i], images[swappos], boardSize );
//                BoardHelper::copyBoard( images[swappos], tempBoard, boardSize );
//            }
//        }
//        for( int i = 0; i < 1000; i++ ) {
//            swap( tempBoard, images, labels, boardSize, i, i + 1000 );
//        }
        BoardHelper::deleteBoard( &tempBoard, boardSize );
    }

    static unsigned int readUInt( unsigned char *data, int location ) {
        unsigned int value = 0;
        for( int i = 0; i < 4; i++ ) {
            int thisbyte = data[location*4+i];
            value += thisbyte << ((3-i) * 8);
        }
        std::cout << "readUint[" << location << "]=" << value << std::endl;
        return value;
    }

    static void writeUInt( unsigned char *data, int location, unsigned int value ) {
        for( int i = 0; i < 4; i++ ) {
            data[location*4+i] = (unsigned char)((value >> ((3-i)*8))&255);
        }
    }
};

