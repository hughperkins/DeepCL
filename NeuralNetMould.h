// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class NeuralNet;

class NeuralNetMould {
public:
    int _numPlanes;
    int _boardSize;
    NeuralNetMould(){
    }
    NeuralNetMould *planes(int planes ) {
        this->_numPlanes = planes;
        return this;
    }
    NeuralNetMould *boardSize( int boardSize ) {
        this->_boardSize = boardSize;
        return this;
    }
    NeuralNet *make();
};


