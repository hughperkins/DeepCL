// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "DeepCLGtestGlobals.h"

#include <iostream>
using namespace std;

DeepCLGtestGlobals *DeepCLGtestGlobals::instance() {
    static DeepCLGtestGlobals *thisInstance = new DeepCLGtestGlobals();
    return thisInstance;
}

easycl::EasyCL *DeepCLGtestGlobals_createEasyCL() {
    int gpuindex = DeepCLGtestGlobals::instance()->gpuindex;
    if(gpuindex == 0) {
        easycl::EasyCL *cl = easycl::EasyCL::createForFirstGpuOtherwiseCpu();
        // cout << "got cl, returning..." << endl;
        return cl;
    } else {
        return easycl::EasyCL::createForIndexedGpu(gpuindex);
    }
}

