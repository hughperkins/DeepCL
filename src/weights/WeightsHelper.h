// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

class WeightsHelper {
public:
    static inline float generateWeight(float rangesize) {
//        float rangesize = sqrt(12.0f / (float)fanin) ;
    //        float uniformrand = random() / (float)random.max();     
        float signeduniformrand = RandomSingleton::uniform() * 2.0f - 1.0f;
        float result = rangesize * signeduniformrand;
//        cout << "generateWeight result=" << result << endl;
        return result;
    }
};

