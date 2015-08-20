// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "RandomTranslations.h"

#include "RandomTranslationsMaker.h"

Layer *RandomTranslationsMaker::createLayer(Layer *previousLayer) {
    return new RandomTranslations(previousLayer, this);
}

