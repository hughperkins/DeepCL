// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class NeuralNet;
class EasyCL;

#include "DeepCLDllExport.h"

class DeepCL_EXPORT NeuralNetMould {
public:
    EasyCL *cl; // NOT delete
    int _numPlanes;
    int _imageSize;
    NeuralNetMould(EasyCL *cl) :
            cl(cl) {
        _numPlanes = 0;
        _imageSize = 0;
    }
    NeuralNetMould(int planes, int imageSize){
        this->_numPlanes = planes;
        this->_imageSize = imageSize;
    }
    NeuralNetMould *planes(int planes) {
        this->_numPlanes = planes;
        return this;
    }
    NeuralNetMould *imageSize(int imageSize) {
        this->_imageSize = imageSize;
        return this;
    }
    NeuralNet *instance();
};


