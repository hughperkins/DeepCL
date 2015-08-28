// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <string>

#include "DeepCLDllExport.h"

class NeuralNet;
class WeightsInitializer;

#define VIRTUAL virtual
#define STATIC static

/// \brief Add layers to a NeuralNet object, based on a netdef-string
///
/// eg "8c5-mp2" will add a convolutional layer with 8 filter, each 
/// 5 by 5; and one max-pooling layer, over 2x2
/// based on the notation proposed in 
/// [Multi-column Deep Neural Networks for Image Classification](http://arxiv.org/pdf/1202.2745.pdf)
PUBLICAPI
class DeepCL_EXPORT NetdefToNet {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    STATIC bool parseSubstring(WeightsInitializer *weightsInitializer, NeuralNet *net, std::string substring, bool isLast);
    PUBLICAPI STATIC bool createNetFromNetdef(NeuralNet *net, std::string netdef);
    PUBLICAPI STATIC bool createNetFromNetdefCharStar(NeuralNet *net, const char *netdef);
    STATIC bool createNetFromNetdef(NeuralNet *net, std::string netdef, WeightsInitializer *weightsInitializer);

    // [[[end]]]
};

