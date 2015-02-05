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

class NormalizationLayerMaker;

class NormalizationLayer : public Layer, IHasToString {
public:
    const float translate; // apply translate first
    const float scale;  // then scale

    const int outputPlanes;
    const int outputBoardSize;

    int batchSize;
    int allocatedSize;
    float *results;

    inline int getResultIndex( int n, int outPlane, int outRow, int outCol ) const {
        return ( ( n
            * outputPlanes + outPlane )
            * outputBoardSize + outRow )
            * outputBoardSize + outCol;
    }
    inline float getResult( int n, int outPlane, int outRow, int outCol ) const {
        return results[ getResultIndex(n,outPlane, outRow, outCol ) ];
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    NormalizationLayer( Layer *previousLayer, NormalizationLayerMaker const*maker );
    VIRTUAL ~NormalizationLayer();
    VIRTUAL float *getResults();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL bool needsBackProp();
    VIRTUAL void printOutput() const;
    VIRTUAL void print() const;
    VIRTUAL bool needErrorsBackprop();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void propagate();
    VIRTUAL void backPropErrors( float learningRate, float const *errors );
    VIRTUAL int getOutputBoardSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getResultsSize() const;
    VIRTUAL std::string toString();
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

std::ostream &operator<<( std::ostream &os, NormalizationLayer &layer );
std::ostream &operator<<( std::ostream &os, NormalizationLayer const*layer );

