// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"

class InputLayer : public Layer {
public:
    InputLayer( int numPlanes, int boardSize ) :
          Layer( 0, numPlanes, boardSize ) {
    }
    virtual void print() {
        if( results == 0 ) {
             return;
        }
        for( int n = 0; n < batchSize; n++ ) {
            std::cout << "InputLayer n " << n << ":" << std::endl;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                if( numPlanes > 1 ) std::cout << "    plane " << plane << ":" << std::endl;
                for( int i = 0; i < std::min(5,boardSize); i++ ) {
                    std::cout << "      ";
                    for( int j = 0; j < std::min(5,boardSize); j++ ) {
                        std::cout << getResult( n, plane, i, j ) << " ";
//results[
//                            n * numPlanes * boardSize*boardSize +
//                            plane*boardSize*boardSize +
//                            i * boardSize +
//                            j ] << " ";
                    }
                    if( boardSize > 5 ) std::cout << " ... ";
                    std::cout << std::endl;
                }
                if( boardSize > 5 ) std::cout << " ... " << std::endl;
            }
        }
    }
    void in( float const*images ) {
//        std::cout << "InputLayer::in()" << std::endl;
        this->results = (float*)images;
//        this->batchStart = batchStart;
//        this->batchEnd = batchEnd;
//        print();
    }
    virtual ~InputLayer() {
    }
    virtual void setBatchSize( int batchSize ) {
        this->batchSize = batchSize;
    }
    virtual void propagate() {
    }
    virtual void backPropErrors( float learningRate, float const *errors ) {
    }
};

