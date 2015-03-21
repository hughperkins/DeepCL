// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class NeuralNet;

#if defined(_WIN32) 
# if defined(ClConvolve_EXPORTS)
#  define ClConvolve_EXPORT __declspec(dllexport)
# else
#  define ClConvolve_EXPORT __declspec(dllimport)
# endif // ClConvolve_EXPORTS
#else // _WIN32
# define ClConvolve_EXPORT
#endif

class ClConvolve_EXPORT NeuralNetMould {
public:
    int _numPlanes;
    int _imageSize;
    NeuralNetMould(){
        _numPlanes = 0;
        _imageSize = 0;
    }
    NeuralNetMould( int planes, int imageSize ){
        this->_numPlanes = planes;
        this->_imageSize = imageSize;
    }
    NeuralNetMould *planes(int planes ) {
        this->_numPlanes = planes;
        return this;
    }
    NeuralNetMould *imageSize( int imageSize ) {
        this->_imageSize = imageSize;
        return this;
    }
    NeuralNet *instance();
};


