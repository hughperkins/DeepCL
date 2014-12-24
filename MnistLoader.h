// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>

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
        int labelsFilesize = 0;
        unsigned char *imagesData = FileHelper::readBinary( dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize );
        unsigned char *labelsData = FileHelper::readBinary( dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize );

        int totalNumImages = readUInt( imagesData, 1 );
        int numRows = readUInt( imagesData, 2 );
        int numCols = readUInt( imagesData, 3 );
        *p_numImages = totalNumImages;
        *p_size = numRows;
        cout << "totalNumImages " << totalNumImages << " " << numRows << "*" << numCols << endl;
        int ***boards = BoardsHelper::allocateBoards( *p_numImages, numRows );
        for( int n = 0; n < *p_numImages; n++ ) {
            for( int i = 0; i < numRows; i++ ) {
                for( int j = 0; j < numRows; j++ ) {
                    boards[n][i][j] = (int)imagesData[16 + n * numRows * numCols + i * numCols + j];
                }
            }
        }
        delete[] imagesData;
        delete[] labelsData;
        return boards;
    }

protected:
    static unsigned int readUInt( unsigned char *data, int location ) {
        unsigned int value = 0;
        for( int i = 0; i < 4; i++ ) {
            int thisbyte = data[location*4+i];
            value += thisbyte << ((3-i) * 8);
        }
        return value;
    }
};

