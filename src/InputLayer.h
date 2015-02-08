// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "ActivationFunction.h"
#include "stringhelper.h"

template<typename T> class InputLayerMaker;

#define VIRTUAL virtual

template< typename T >
class InputLayer : public Layer, IHasToString {
public:
    int batchSize;
    int allocatedSize;

    const int outputPlanes;
    const int outputBoardSize;

    T const*input; // we dont own this
    float *results; // we own this :-)

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
    // cog_addheaders.add_templated()
    // ]]]
    // generated, using cog:
    InputLayer( InputLayerMaker<T> *maker );
    VIRTUAL ~InputLayer();
    VIRTUAL float *getResults();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getPersistSize() const;
    VIRTUAL void printOutput() const;
    VIRTUAL void print() const;
    void in( T const*images );
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

template< typename T > std::ostream &operator<<( std::ostream &os, InputLayer<T> &layer );
template< typename T > std::ostream &operator<<( std::ostream &os, InputLayer<T> const*layer );

