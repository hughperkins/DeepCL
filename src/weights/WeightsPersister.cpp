// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstring>

#include "util/FileHelper.h"
#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include "weights/WeightsPersister.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

template< typename T > STATIC void WeightsPersister::copyArray(T *dst, T const*src, int length) { // this might already be in standard C++ library?
    memcpy(dst, src, length * sizeof(T));
}
STATIC int WeightsPersister::getTotalNumWeights(NeuralNet *net) {
    return getTotalNumWeights(latestVersion, net);
}
STATIC int WeightsPersister::getTotalNumWeights(int version, NeuralNet *net) {
    int totalWeightsSize = 0;
//    cout << "layers size " << net->layers.size() << endl;
    for(int layerIdx = 1; layerIdx < net->getNumLayers(); layerIdx++) {
        Layer *layer = net->getLayer(layerIdx);
        int thisPersistSize = layer->getPersistSize(version);
//        cout << "layer " << layerIdx << " this persist size " << thisPersistSize << endl;
        totalWeightsSize += thisPersistSize;
    }
    return totalWeightsSize;
}
STATIC void WeightsPersister::copyNetWeightsToArray(NeuralNet *net, float *target) {
    copyNetWeightsToArray(latestVersion, net, target);
}
STATIC void WeightsPersister::copyNetWeightsToArray(int version, NeuralNet *net, float *target) {
    int pos = 0;
    for(int layerIdx = 1; layerIdx < net->getNumLayers(); layerIdx++) {
        Layer *layer = net->getLayer(layerIdx);
        int persistSize = layer->getPersistSize(version);
        if(persistSize > 0) {
            layer->persistToArray(version, &(target[pos]));
        }
        pos += persistSize;
    }
}
STATIC void WeightsPersister::copyArrayToNetWeights(float const*source, NeuralNet *net) {
    copyArrayToNetWeights(latestVersion, source, net);
}
STATIC void WeightsPersister::copyArrayToNetWeights(int version, float const*source, NeuralNet *net) {
    int pos = 0;
    for(int layerIdx = 1; layerIdx < net->getNumLayers(); layerIdx++) {
    Layer *layer = net->getLayer(layerIdx);
        int persistSize = layer->getPersistSize(version);
        if(persistSize > 0) {
            layer->unpersistFromArray(version, &(source[pos]));
        }
        pos += persistSize;
    }
}
STATIC int WeightsPersister::getArrayOffsetForLayer(NeuralNet *net, int layer) {
    return getArrayOffsetForLayer(latestVersion, net, layer);
}
STATIC int WeightsPersister::getArrayOffsetForLayer(int version, NeuralNet *net, int layer) {
    int pos = 0;
    for(int layerIdx = 1; layerIdx < layer; layerIdx++) {
    Layer *layer = net->getLayer(layerIdx);
        pos += layer->getPersistSize(version);
    }
    return pos;
}
// this will either succeed or fail in general
// in the worst case, you can find the weights in a file postfixed with '~', if 
// the machine fails right in between the 'delete' and the 'rename', but you 
// should ideally never actually lose the weights file (unless the drive itself
// fails of course...)
STATIC void WeightsPersister::persistWeights(std::string filepath, std::string trainingConfigString, NeuralNet *net, int epoch, int batch, float annealedLearningRate, int numRight, float loss) { // we should probably rename 'weights' to 'model' now that we are storing normalization data too?
    int headerLength = 1024;
    int totalWeightsSize = getTotalNumWeights(latestVersion, net);
    char *persistArray = new char[headerLength + totalWeightsSize * sizeof(float) ];
    int *persistArrayInts = reinterpret_cast<int *>(persistArray);
    float *persistArrayFloats = reinterpret_cast<float *>(persistArray);
    strcpy_safe(persistArray, "ClCn", 4); // so easy to recognise file type
    persistArrayInts[1] = latestVersion; // data file version number
    persistArrayInts[2] = epoch;
    persistArrayInts[3] = batch;
    persistArrayInts[4] = numRight;
    persistArrayFloats[5] = loss;
    persistArrayFloats[6] = annealedLearningRate;
    strcpy_safe(persistArray + 7 * 4, trainingConfigString.c_str(), 800);
    copyNetWeightsToArray(latestVersion, net, reinterpret_cast<float *>(persistArray + headerLength));
    FileHelper::writeBinary(filepath + "~", reinterpret_cast<char *>(persistArray), 
        headerLength + totalWeightsSize * sizeof(float));
    FileHelper::remove(filepath);
    FileHelper::rename(filepath + "~", filepath);
    std::cout << "wrote weights to file, filesize " << (( headerLength + totalWeightsSize) *sizeof(float)/1024) << "KB" << std::endl;
    delete[] persistArray;
}
STATIC bool WeightsPersister::loadWeights(std::string filepath, std::string trainingConfigString, NeuralNet *net, int *p_epoch, int *p_batch, float *p_annealedLearningRate, int *p_numRight, float *p_loss) {
    if(FileHelper::exists(filepath) ){
        int headerSize = 1024;
        long fileSize;
        char * data = FileHelper::readBinary(filepath, &fileSize);

        if(!checkData(data, headerSize, fileSize) ){
            delete [] data;
            return false;
        }
        int *dataAsInts = reinterpret_cast<int *>(data);
        int version = dataAsInts[1];
        if(version == 1 || version == 3) {
            return loadWeightsv1or3(data, fileSize, trainingConfigString, net, p_epoch, p_batch, p_annealedLearningRate, p_numRight, p_loss);
        } else {
            throw std::runtime_error("weights version " + toString(version) + " not recognized");
        }
    }
    return false;
}
STATIC bool WeightsPersister::loadWeightsv1or3(char *data, long fileSize, std::string trainingConfigString, NeuralNet *net, int *p_epoch, int *p_batch, float *p_annealedLearningRate, int *p_numRight, float *p_loss) {
        int headerSize = 1024;
        data[headerSize - 1] = 0; // null-terminate the string, if not already done

        if(trainingConfigString != std::string(data + 7 * 4)) {
            std::cout << "training options dont match weights file" << std::endl;
            std::cout << "in file: [" + std::string(data + 7 * 4) + "]" << std::endl;
            std::cout << "current options: [" + trainingConfigString + "]" << std::endl;

            delete [] data;
            return false;
        }
        
        int *dataAsInts = reinterpret_cast<int *>(data);
        float *dataAsFloats = reinterpret_cast<float *>(data);
        float *allWeightsArray = reinterpret_cast<float *>(data + headerSize);
        int version = dataAsInts[1];
        if(version == 1 || version == 3) {
            *p_epoch = dataAsInts[2];
            *p_batch = dataAsInts[3];
            *p_numRight = dataAsInts[4];
            *p_loss = dataAsFloats[5];
            *p_annealedLearningRate = dataAsFloats[6];
        } else {
            throw runtime_error("Unrecognized version " + toString(version) );
        }

//        std::cout << "read weights from file "  << (fileSize/1024) << "KB" << std::endl;
        int expectedTotalWeightsSize = getTotalNumWeights(version, net);
        int numFloatsRead = (fileSize - headerSize) / sizeof(float);

        if(expectedTotalWeightsSize != numFloatsRead) {
            delete [] data;
            throw std::runtime_error("weights file contains " + toString(numFloatsRead) + " floats, but we expect to see: " + toString(expectedTotalWeightsSize) + ".  So there is probably some mismatch between the weights file, and the settings, or network version, used.");
        }

        copyArrayToNetWeights(version, allWeightsArray, net);

        delete [] data;
        return true;
}
STATIC bool WeightsPersister::checkData(const char * data, long headerSize, long fileSize) {
    if(fileSize < headerSize) {
        std::cout << "weights file has invalid size" << std::endl;
        return false;
    }

    if(data[0] != 'C' || data[1] != 'l' || data[2] != 'C' || data[3] != 'n') {
        std::cout << "weights file not ClConvolve format" << std::endl;
        return false;
    }

    const int *dataAsInts = reinterpret_cast<const int *>(data);
    if(dataAsInts[1] != 1 && dataAsInts[1] != 3) {
        std::cout << "weights file version not known" << std::endl;
        return false;
    }

    return true;
}
STATIC bool WeightsPersister::loadConfigString(std::string filepath, std::string & configString) {
    if(FileHelper::exists(filepath) ){
        int headerSize = 1024;
        long fileSize;
        char * data = FileHelper::readBinary(filepath, &fileSize);

        if(!checkData(data, headerSize, fileSize) ) {
            delete [] data;
            return false;
        }

        data[headerSize - 1] = 0; // null-terminate the string, if not already done

        // + skip the 'netdef='
        const int *dataAsInts = reinterpret_cast<const int *>(data);
        int version = dataAsInts[1];
        if(version == 1 || version == 3) {
            configString = std::string(data + 7 * 4 + 7);
        } else {
            throw std::runtime_error("unknown versoin " + toString(version));
        }

        delete [] data;
        return true;
    }
    return false;
}
