// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <string>

class NeuralNet;

#define VIRTUAL virtual
#define STATIC static

#include "DeepCLDllExport.h"

/// \brief Use to read/write weights from a NeuralNet
///
/// whilst this class is portable, the weights files created totally are not (ie: endianness)
/// but okish for now... (since it's not like weights files tend to be shared around much, and
/// if they are, then the quickly-written file created by this could be converted by another
/// utility into a portable datafile
///
/// Target usage for this class is quickly snapshotting the weights after each epoch.  
/// Therefore should be: fast, low IO :-)
/// 
PUBLICAPI
class DeepCL_EXPORT WeightsPersister {
public:
    static const int latestVersion = 3;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    template< typename T > STATIC void copyArray(T *dst, T const*src, int length);  // this might already be in standard C++ library?
    STATIC int getTotalNumWeights(NeuralNet *net);
    STATIC int getTotalNumWeights(int version, NeuralNet *net);
    STATIC void copyNetWeightsToArray(NeuralNet *net, float *target);
    STATIC void copyNetWeightsToArray(int version, NeuralNet *net, float *target);
    STATIC void copyArrayToNetWeights(float const*source, NeuralNet *net);
    STATIC void copyArrayToNetWeights(int version, float const*source, NeuralNet *net);
    STATIC int getArrayOffsetForLayer(NeuralNet *net, int layer);
    STATIC int getArrayOffsetForLayer(int version, NeuralNet *net, int layer);
    STATIC void persistWeights(std::string filepath, std::string trainingConfigString, NeuralNet *net, int epoch, int batch, float annealedLearningRate, int numRight, float loss);  // we should probably rename 'weights' to 'model' now that we are storing normalization data too?
    STATIC bool loadWeights(std::string filepath, std::string trainingConfigString, NeuralNet *net, int *p_epoch, int *p_batch, float *p_annealedLearningRate, int *p_numRight, float *p_loss);
    STATIC bool loadWeightsv1or3(char *data, long fileSize, std::string trainingConfigString, NeuralNet *net, int *p_epoch, int *p_batch, float *p_annealedLearningRate, int *p_numRight, float *p_loss);
    STATIC bool checkData(const char * data, long headerSize, long fileSize);
    STATIC bool loadConfigString(std::string filepath, std::string & configString);

    // [[[end]]]
};

