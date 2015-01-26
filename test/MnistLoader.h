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
    static int **loadImage( std::string dir, std::string set, int idx, int *p_size ) {
        long imagesFilesize = 0;
        long labelsFilesize = 0;
        char *imagesDataSigned = FileHelper::readBinary( dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize );
        char *labelsDataSigned = FileHelper::readBinary( dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize );
        unsigned char *imagesData = reinterpret_cast< unsigned char *>(imagesDataSigned);
        unsigned char *labelsData = reinterpret_cast< unsigned char *>(labelsDataSigned);

        int numImages = readUInt( imagesData, 1 );
        int numRows = readUInt( imagesData, 2 );
        int numCols = readUInt( imagesData, 3 );
        *p_size = numRows;
        std::cout << "numimages " << numImages << " " << numRows << "*" << numCols << std::endl;

        int **board = BoardHelper::allocateBoard( numRows );
        for( int i = 0; i < numRows; i++ ) {
            for( int j = 0; j < numRows; j++ ) {
                board[i][j] = (int)imagesData[idx * numRows * numCols + i * numCols + j];
            }
        }
        delete[] imagesDataSigned;
        delete[] labelsDataSigned;
        return board;
    }
    static int ***loadImages( std::string dir, std::string set, int *p_numImages, int *p_size ) {
        long imagesFilesize = 0;
        char *imagesDataSigned = FileHelper::readBinary( dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize );
        unsigned char *imagesData = reinterpret_cast<unsigned char *>(imagesDataSigned);
        int totalNumImages = readUInt( imagesData, 1 );
        int numRows = readUInt( imagesData, 2 );
        int numCols = readUInt( imagesData, 3 );
//        *p_numImages = min(100,totalNumImages);
        *p_numImages = totalNumImages;
        *p_size = numRows;
        std::cout << "totalNumImages " << *p_numImages << " " << *p_size << "*" << numCols << std::endl;
        int ***boards = BoardsHelper::allocateBoards( *p_numImages, numRows );
        for( int n = 0; n < *p_numImages; n++ ) {
            for( int i = 0; i < numRows; i++ ) {
                for( int j = 0; j < numRows; j++ ) {
                    boards[n][i][j] = (int)imagesData[16 + n * numRows * numCols + i * numCols + j];
                }
            }
        }
        delete[] imagesDataSigned;
        return boards;
    }
    static int *loadLabels( std::string dir, std::string set, int *p_numImages ) {
        long labelsFilesize = 0;
        char *labelsDataSigned = FileHelper::readBinary( dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize );
        unsigned char *labelsData = reinterpret_cast<unsigned char *>(labelsDataSigned);
        int totalNumImages = readUInt( labelsData, 1 );
      //  *p_numImages = min(100,totalNumImages);
        *p_numImages = totalNumImages;
        std::cout << "set " << set << " num labels " << *p_numImages << std::endl;
        int *labels = new int[*p_numImages];
        for( int n = 0; n < *p_numImages; n++ ) {
           labels[n] = (int)labelsData[8 + n];
        }
        delete[] labelsDataSigned;
        return labels;
    }

    static int readUInt( unsigned char *data, int location ) {
        unsigned int value = 0;
        for( int i = 0; i < 4; i++ ) {
            int thisbyte = data[location*4+i];
            value += thisbyte << ((3-i) * 8);
        }
        std::cout << "readUint[" << location << "]=" << value << std::endl;
        return value;
    }

    static void writeUInt( unsigned char *data, int location, int value ) {
        for( int i = 0; i < 4; i++ ) {
            data[location*4+i] = ((value >> ((3-i)*8))&255);
        }
    }
};

