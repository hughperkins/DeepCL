// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>

#include "DeepCLDllExport.h"

class DeepCL_EXPORT Statistics {
public:
    int count;
    float maxY;
    float minY;
    float sumY;
    float sumYSquared;
    Statistics() {
        memset(this, 0, sizeof(Statistics) );
    }
};

class DeepCL_EXPORT NormalizationHelper {
public:
    static void updateStatistics(float *Y, int length, int cubeSize, Statistics *statistics) {
        float thisSumY = 0;
        float thisSumYSquared = 0;
        float thisMin = Y[0];
        float thisMax = Y[0];
        for(int i = 0; i < length * cubeSize; i++) {
            float thisValue = Y[i];
            thisSumY += thisValue;
            thisSumYSquared += (float)thisValue * (float)thisValue;
            thisMin = thisValue < thisMin ? thisValue : thisMin;
            thisMax = thisValue > thisMax ? thisValue : thisMax;
//            std::cout << "Y[i] " << (int)Y[i] << std::endl;
//            std::cout << "updatestatistics " << i << " thissumy=" << thisSumY << " thisSumYSquared=" << thisSumYSquared << std::endl;
        }
        statistics->count += length * cubeSize;
        statistics->maxY = thisMax > statistics->maxY ? thisMax : statistics->maxY;
        statistics->minY = thisMin < statistics->minY ? thisMin : statistics->minY;
        statistics->sumY += thisSumY;
        statistics->sumYSquared += thisSumYSquared;
    }

    static void calcMeanAndStdDev(Statistics *statistics, float *p_mean, float *p_stdDev) {
        *p_mean = (float)statistics->sumY / statistics->count;
        *p_stdDev = sqrt(( statistics->sumYSquared - statistics->sumY * statistics->sumY / statistics->count) / (statistics->count - 1) );
    }
    
    static void getMeanAndStdDev(float *data, int length, float *p_mean, float *p_stdDev) {
        // get mean of the dataset, and stddev
    //    float thismax = 0;
        float sum = 0;
        for(int i = 0; i < length; i++) {
            float thisValue = data[i];
            sum += thisValue;
        }
        float mean = sum / length;

        float sumSquaredDiff = 0;
        for(int i = 0; i < length; i++) {
            float thisValue = data[i];
//            std::cout << "i " << i << "=" << thisValue << std::endl;
            float diffFromMean = thisValue - mean;
            float diffSquared = diffFromMean * diffFromMean;
            sumSquaredDiff += diffSquared;
        }
        float stdDev = (float)std::sqrt((double)(sumSquaredDiff / (length - 1)));

        *p_mean = mean;
        *p_stdDev = stdDev;
    }
    
    static void getMeanAndMaxDev(float *data, int length, float *p_mean, float *p_maxDev) {
        // get mean of the dataset, and stddev
    //    float thismax = 0;
        float sum = 0;
        for(int i = 0; i < length; i++) {
            float thisValue = data[i];
            sum += thisValue;
        }
        float mean = sum / length;

//        float sumSquaredDiff = 0;
//        for(int i = 0; i < length; i++) {
//            int thisValue = (int)data[i];
//            float diffFromMean = thisValue - mean;
//            float diffSquared = diffFromMean * diffFromMean;
//            sumSquaredDiff += diffSquared;
//        }
//        float stdDev = sqrt(sumSquaredDiff / (length - 1) );

        *p_mean = mean;
        *p_maxDev = std::max<float>(255-mean, mean);
    }
    
    static void getMinMax(float *data, int length, float *p_middle, float *p_maxDev) {
        // get mean of the dataset, and stddev
        float thismin = 0;
        float thismax = 0;
//        float sum = 0;
        for(int i = 0; i < length; i++) {
            float thisValue = data[i];
            thismin = std::min<float>(thisValue, thismin);
            thismax = std::max<float>(thisValue, thismax);
        }

        *p_middle = (thismax + thismin) / 2; // pick number in the middle
        *p_maxDev = (thismax - thismin) / 2; // distance from middle of range to either end
    }
    
    static void normalize(float *data, int length, float mean, float scaling) {
        for(int i = 0; i < length; i++) {
            data[i] = (data[i] - mean) / scaling;
        }
    }
};



