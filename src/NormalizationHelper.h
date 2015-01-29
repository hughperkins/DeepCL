// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>

class NormalizationHelper {
public:
    template<typename T>
    static void getMeanAndStdDev( T *data, int length, float *p_mean, float *p_stdDev ) {
        // get mean of the dataset, and stddev
    //    float thismax = 0;
        float sum = 0;
        for( int i = 0; i < length; i++ ) {
            float thisValue = data[i];
            sum += thisValue;
        }
        float mean = sum / length;

        float sumSquaredDiff = 0;
        for( int i = 0; i < length; i++ ) {
            float thisValue = data[i];
//            std::cout << "i " << i << "=" << thisValue << std::endl;
            float diffFromMean = thisValue - mean;
            float diffSquared = diffFromMean * diffFromMean;
            sumSquaredDiff += diffSquared;
        }
        float stdDev = sqrt( sumSquaredDiff / ( length - 1 ) );

        *p_mean = mean;
        *p_stdDev = stdDev;
    }

    template<typename T>
    static void getMeanAndMaxDev( T *data, int length, float *p_mean, float *p_maxDev ) {
        // get mean of the dataset, and stddev
    //    float thismax = 0;
        float sum = 0;
        for( int i = 0; i < length; i++ ) {
            float thisValue = data[i];
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
        *p_maxDev = std::max<float>( 255-mean, mean );
    }

    template<typename T>
    static void getMinMax( T *data, int length, float *p_middle, float *p_maxDev ) {
        // get mean of the dataset, and stddev
        float thismin = 0;
        float thismax = 0;
        float sum = 0;
        for( int i = 0; i < length; i++ ) {
            float thisValue = data[i];
            thismin = std::min<float>( thisValue, thismin );
            thismax = std::max<float>( thisValue, thismax );
        }

        *p_mean = ( thismax + thismin ) / 2; // pick number in the middle
        *p_maxDev = ( thismax - thismin ) / 2; // distance from middle of range to either end
    }

    template<typename T>
    static void normalize( T *data, int length, float mean, float scaling ) {
        for( int i = 0; i < length; i++ ) {
            data[i] = ( data[i] - mean ) / scaling;
        }
    }
};

