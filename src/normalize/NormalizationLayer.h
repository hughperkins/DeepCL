// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "layer/Layer.h"
#include "activate/ActivationFunction.h"
#include "util/stringhelper.h"

#define VIRTUAL virtual

class NormalizationLayerMaker;

class NormalizationLayer : public Layer, IHasToString {
public:
    float translate; // apply translate first
    float scale;  // then scale

    const int outputPlanes;
    const int outputSize;

    int batchSize;
    int allocatedSize;
    float *output;

    inline int getResultIndex(int n, int outPlane, int outRow, int outCol) const {
        return (( n
            * outputPlanes + outPlane)
            * outputSize + outRow)
            * outputSize + outCol;
    }
    inline float getResult(int n, int outPlane, int outRow, int outCol) const {
        return output[ getResultIndex(n,outPlane, outRow, outCol) ];
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    NormalizationLayer(Layer *previousLayer, NormalizationLayerMaker *maker);
    VIRTUAL ~NormalizationLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float *getOutput();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL void persistToArray(int version, float *array);
    VIRTUAL void unpersistFromArray(int version, float const*array);
    VIRTUAL bool needsBackProp();
    VIRTUAL void printOutput() const;
    VIRTUAL void print() const;
    VIRTUAL bool needErrorsBackprop();
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL void forward();
    VIRTUAL void backward(float learningRate, float const *gradOutput);
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL std::string toString();
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

std::ostream &operator<<(std::ostream &os, NormalizationLayer &layer);
std::ostream &operator<<(std::ostream &os, NormalizationLayer const*layer);

