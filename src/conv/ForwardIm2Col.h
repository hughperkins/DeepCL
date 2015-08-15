// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Forward.h"

class AddBias;

class ForwardIm2Col : public Forward {
private:
    CLKernel *kernel;
    AddBias *addBias;
	float *columns;
	float *ones;
	CLWrapper *columnsWrapper;
	CLWrapper *onesWrapper;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:
    VIRTUAL ~ForwardIm2Col();
    VIRTUAL void forward( int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper );
    ForwardIm2Col( EasyCL *cl, LayerDimensions dim );

    // [[[end]]]
};

