// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Layer.h"
#include "ActivationFunction.h"
#include "stringhelper.h"

class InputLayerMaker;

#define VIRTUAL virtual

class InputLayer : public Layer, IHasToString {
public:
    int batchSize;
    int allocatedSize;

    const int outputPlanes;
    const int outputImageSize;

    float const*input; // we dont own this
    float *output; // we own this :-)

    inline int getOutputIndex( int n, int outPlane, int outRow, int outCol ) const {
        return ( ( n
            * outputPlanes + outPlane )
            * outputImageSize + outRow )
            * outputImageSize + outCol;
    }
    inline float getOutput( int n, int outPlane, int outRow, int outCol ) const {
        return output[ getOutputIndex(n,outPlane, outRow, outCol ) ];
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    InputLayer( InputLayerMaker *maker );
    VIRTUAL ~InputLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getPersistSize() const;
    VIRTUAL void printOutput();
    VIRTUAL void print();
    void in( float const*images );
    VIRTUAL bool needErrorsBackprop();
    VIRTUAL void setBatchSize( int batchSize );
    VIRTUAL void forward();
    VIRTUAL void backward( float learningRate, float const *gradOutput );
    VIRTUAL int getOutputImageSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputSize() const;
    VIRTUAL std::string toString();
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

 std::ostream &operator<<( std::ostream &os, InputLayer &layer );
 std::ostream &operator<<( std::ostream &os, InputLayer const*layer );

