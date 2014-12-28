// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class AccuracyHelper{
public:
    static int calcNumRight( int numImages, int numPlanes, int const*labels, float const*results ) {
        int correct = 0;
        for( int n = 0; n < numImages; n++ ) {
            double maxValue = -100000;
            int bestIndex = -1;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                if( results[ n * numPlanes + plane ] > maxValue ) {
                    bestIndex = plane;
                    maxValue = results[ n * numPlanes + plane ];
                }
            }
//            cout << "expected: " << labels[n] << " got " << bestIndex << endl;
            if( bestIndex == labels[n] ) {
                correct++;
            }
        }
        return correct;
    }
    static void printAccuracy( int numImages, int numPlanes, int const*labels, float const*results ) {
        int correct = 0;
        for( int n = 0; n < numImages; n++ ) {
            double maxValue = -100000;
            int bestIndex = -1;
            for( int plane = 0; plane < numPlanes; plane++ ) {
                if( results[ n * numPlanes + plane ] > maxValue ) {
                    bestIndex = plane;
                    maxValue = results[ n * numPlanes + plane ];
                }
            }
//            cout << "expected: " << labels[n] << " got " << bestIndex << endl;
            if( bestIndex == labels[n] ) {
                correct++;
            }
        }
        std::cout << " accuracy: " << correct << "/" << numImages << std::endl;
    }
};

