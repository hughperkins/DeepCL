// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class NormalizationHelper {
public:
    template<typename T>
    static void getStats( T *data, int length, float *p_mean, float *p_scaling ) {
        // get mean of the dataset, and stddev
    //    float thismax = 0;
        float sum = 0;
        for( int i = 0; i < length; i++ ) {
            int thisValue = (int)data[i];
            sum += thisValue;
        }
        float mean = sum / length;

//        float sumSquaredDiff = 0;
//        for( int i = 0; i < length; i++ ) {
//            int thisValue = (int)data[i];
//            float diffFromMean = thisValue - mean;
//            float diffSquared = diffFromMean * diffFromMean;
//            sumSquaredDiff += diffSquared;
//        }
//        float stdDev = sqrt( sumSquaredDiff / ( length - 1 ) );

        *p_mean = mean;
        *p_scaling = std::max<float>( 255-mean, mean );
    }

    template<typename T>
    static void normalize( T *data, int length, double mean, double scaling ) {
        for( int i = 0; i < length; i++ ) {
            data[i] = ( data[i] - mean ) / scaling;
        }
    }
};

