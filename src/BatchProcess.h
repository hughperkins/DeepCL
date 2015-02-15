// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "NormalizationHelper.h"

#include "ClConvolveDllExport.h"

template< typename T >
class BatchAction {
public:
    T *data;
    int *labels;
    BatchAction( T *data, int *labels ) :
        data(data),
        labels(labels) { // have to provide appropriate buffers for this
    }
    virtual void processBatch( int batchSize, int cubeSize ) = 0;
};

class ClConvolve_EXPORT BatchProcess {
public:
    template< typename T>
    static void run(std::string filepath, int startN, int batchSize, int totalN, int cubeSize, BatchAction<T> *batchAction);
};

template< typename T >
class NormalizeGetStdDev : public BatchAction<T> {
public:
    Statistics<T> statistics; 
    NormalizeGetStdDev( T *data, int *labels ) :
        BatchAction<T>::BatchAction( data, labels ) {
    }
    virtual void processBatch( int batchSize, int cubeSize ) {
        NormalizationHelper::updateStatistics( this->data, batchSize, cubeSize, &statistics );
    }
    void calcMeanStdDev( float *p_mean, float *p_stdDev ) {
        NormalizationHelper::calcMeanAndStdDev( &statistics, p_mean, p_stdDev );
    }
};

template< typename T >
class NormalizeGetMinMax : public BatchAction<T> {
public:
    Statistics<T> statistics; 
    NormalizeGetMinMax( T *data, int *labels ) :
        BatchAction<T>( data, labels ) {
    }
    virtual void processBatch( int batchSize, int cubeSize ) {
        NormalizationHelper::updateStatistics( this->data, batchSize, cubeSize, &statistics );
    }
    void calcMinMaxTransform( float *p_translate, float *p_scale ) {
        // add this to our values to center
        *p_translate = - ( statistics.maxY - statistics.minY ) / 2.0f;
        // multiply our values by this to scale to -1 / +1 range
        *p_scale = 1.0f / ( statistics.maxY - statistics.minY );
    }
};

