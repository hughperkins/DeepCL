// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <stdexcept>
using namespace std;

#include "BoardsHelper.h"

class MnistLoader {
public:
    static unsigned int readUInt( unsigned char *data, int location ) {
        unsigned int value = 0;
        for( int i = 0; i < 4; i++ ) {
            int thisbyte = data[location*4+i];
    //        cout << i << " thisbytes " << thisbyte << endl;
            value += thisbyte << ((3-i) * 8);
    //        value += ((int)data[location*4+i]) << (i * 8);
        }
    //    cout << value << endl;
        return value;
    //    return ( (int)data[location * 4 + 3] << 0 ) + ( (int)data[location * 4 + 2] << 8 ) + 
    //         ( (int)data[location * 4 + 1] << 16 ) + ( (int)data[location * 4 + 0] << 24 );
    }

    static unsigned char *readBinary( string filepath, int *p_filesize ) {
        ifstream file( filepath.c_str(), ios::in | ios::binary | ios::ate);
        if(!file.is_open()) {
            throw runtime_error(filepath);
        }
        *p_filesize = (int)file.tellg();
    //    cout << filepath << " filesize " << *p_filesize << endl;
        char *data = new char[*p_filesize];
        file.seekg(0, ios::beg);
        if(!file.read( data, *p_filesize )) {
            throw runtime_error("failed to read from " + filepath );
        }
        file.close();
        return (unsigned char *)data;
    }

    static int **loadImage( string dir, string set, int idx, int *p_size ) {
        int imagesFilesize = 0;
        int labelsFilesize = 0;
        unsigned char *imagesData = readBinary( dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize );
        unsigned char *labelsData = readBinary( dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize );

    //    cout << "magicNumber " << (int)data[0] << (int)data[1] << (int)data[2] << (int)data[3] << endl;
    //    int type = (int)data[3];
    //    if( type == 3 ) {
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
    static int ***loadImages( string dir, string set, int numImages, int *p_size ) {
        int imagesFilesize = 0;
        int labelsFilesize = 0;
        unsigned char *imagesData = readBinary( dir + "/" + set + "-images-idx3-ubyte", &imagesFilesize );
        unsigned char *labelsData = readBinary( dir + "/" + set + "-labels-idx1-ubyte", &labelsFilesize );

    //    cout << "magicNumber " << (int)data[0] << (int)data[1] << (int)data[2] << (int)data[3] << endl;
    //    int type = (int)data[3];
    //    if( type == 3 ) {
        int totalNumImages = readUInt( imagesData, 1 );
        int numRows = readUInt( imagesData, 2 );
        int numCols = readUInt( imagesData, 3 );
        *p_size = numRows;
        cout << "totalNumImages " << totalNumImages << " " << numRows << "*" << numCols << endl;
        int ***boards = BoardsHelper::allocateBoards( numImages, numRows );
        for( int n = 0; n < numImages; n++ ) {
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
};

