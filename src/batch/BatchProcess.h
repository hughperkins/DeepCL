// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "normalize/NormalizationHelper.h"

#include "DeepCLDllExport.h"

class GenericLoaderv2;

class DeepCL_EXPORT BatchAction {
public:
    float *data;
    int *labels;
    BatchAction(float *data, int *labels) :
        data(data),
        labels(labels) { // have to provide appropriate buffers for this
    }
    virtual void processBatch(int batchSize, int cubeSize) = 0;
};

class DeepCL_EXPORT BatchProcessv2 {
public:
    static void run(GenericLoaderv2*loader, int startN, int batchSize, int totalN, int cubeSize, BatchAction *batchAction);
};

class DeepCL_EXPORT BatchProcess {
public:
    static void run(std::string filepath, int startN, int batchSize, int totalN, int cubeSize, BatchAction *batchAction);
};

class DeepCL_EXPORT NormalizeGetStdDev : public BatchAction {
public:
    Statistics statistics; 
    NormalizeGetStdDev(float *data, int *labels) :
        BatchAction(data, labels) {
    }
    virtual void processBatch(int batchSize, int cubeSize) {
        NormalizationHelper::updateStatistics(this->data, batchSize, cubeSize, &statistics);
    }
    void calcMeanStdDev(float *p_mean, float *p_stdDev) {
        NormalizationHelper::calcMeanAndStdDev(&statistics, p_mean, p_stdDev);
    }
};


class DeepCL_EXPORT NormalizeGetMinMax : public BatchAction {
public:
    Statistics statistics; 
    NormalizeGetMinMax(float *data, int *labels) :
        BatchAction(data, labels) {
    }
    virtual void processBatch(int batchSize, int cubeSize) {
        NormalizationHelper::updateStatistics(this->data, batchSize, cubeSize, &statistics);
    }
    void calcMinMaxTransform(float *p_translate, float *p_scale) {
        // add this to our values to center
        *p_translate = - (statistics.maxY - statistics.minY) / 2.0f;
        // multiply our values by this to scale to -1 / +1 range
        *p_scale = 1.0f / (statistics.maxY - statistics.minY);
    }
};

