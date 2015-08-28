// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "loaders/GenericLoader.h"
#include "loaders/GenericLoaderv2.h"

#include "BatchProcess.h"

#include "DeepCLDllExport.h"

using namespace std;

void BatchProcess::run(std::string filepath, int startN, int batchSize, int totalN, int cubeSize, BatchAction *batchAction) {
    int numBatches = (totalN + batchSize - 1) / batchSize;
    int thisBatchSize = batchSize;
//    cout << "batchProcess::run batchsize " << batchSize << " startN " << startN << " totalN " << totalN << " numBatches " << numBatches << endl;
    for(int batch = 0; batch < numBatches; batch++) {
        int batchStart = batch * batchSize;
        if(batch == numBatches - 1) {
            thisBatchSize = totalN - batchStart;
//            cout << "size of last batch: " << thisBatchSize << endl;
        }
//        cout << "   batchStart " << batchStart << " thisBatchSize " << thisBatchSize << endl;
        GenericLoader::load(filepath.c_str(), batchAction->data, batchAction->labels, batchStart, thisBatchSize);
        batchAction->processBatch(thisBatchSize, cubeSize);
    }
}

void BatchProcessv2::run(GenericLoaderv2 *loader, int startN, int batchSize, int totalN, int cubeSize, BatchAction *batchAction) {
    int numBatches = (totalN + batchSize - 1) / batchSize;
    int thisBatchSize = batchSize;
//    cout << "batchProcess::run batchsize " << batchSize << " startN " << startN << " totalN " << totalN << " numBatches " << numBatches << endl;
    for(int batch = 0; batch < numBatches; batch++) {
        int batchStart = batch * batchSize;
        if(batch == numBatches - 1) {
            thisBatchSize = totalN - batchStart;
//            cout << "size of last batch: " << thisBatchSize << endl;
        }
//        cout << "   batchStart " << batchStart << " thisBatchSize " << thisBatchSize << endl;
        loader->load(batchAction->data, batchAction->labels, batchStart, thisBatchSize);
        batchAction->processBatch(thisBatchSize, cubeSize);
    }
}

