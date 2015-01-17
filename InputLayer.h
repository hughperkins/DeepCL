// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "ActivationFunction.h"
#include "stringhelper.h"

#define VIRTUAL virtual

class InputLayer : public Layer, IHasToString {
public:
    int batchSize;
    float *output; // we dont own this
    const int outputPlanes;
    const int outputBoardSize;

    inline int getResultIndex( int n, int outPlane, int outRow, int outCol ) const {
        return ( ( n
            * outputPlanes + outPlane )
            * outputBoardSize + outRow )
            * outputBoardSize + outCol;
    }
    inline float getResult( int n, int outPlane, int outRow, int outCol ) const {
        return output[ getResultIndex(n,outPlane, outRow, outCol ) ];
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // classname: InputLayer
    // cppfile: InputLayer.cpp

    InputLayer( Layer *previousLayer, InputLayerMaker const*maker );
    VIRTUAL ~InputLayer();
    VIRTUAL float *getResults();
    VIRTUAL void printOutput() const;
    VIRTUAL void print() const;
    void in( float const*images );
    VIRTUAL bool needErrorsBackprop();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void propagate();
    VIRTUAL void backPropErrors( float learningRate, float const *errors );
    VIRTUAL int getOutputBoardSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getResultsSize() const;
    VIRTUAL std::string toString();

    // [[[end]]]
};

std::ostream &operator<<( std::ostream &os, InputLayer &layer );
std::ostream &operator<<( std::ostream &os, InputLayer const*layer );

