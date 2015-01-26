#pragma once

#include <iostream>

#include "FileHelper.h"

// whilst this class is portable, the weights files created totally are not (ie: endianness)
// but okish for now... (since it's not like weights files tend to be shared around much, and
// if they are, then the quickly-written file created by this could be converted by another
// utility into a portable datafile
// target usage for this class:
// - quickly snapshotting the weights after each epoch, therefore should be:
//    - fast, low IO :-)
class WeightsPersister {
public:
    template<typename T>
    static void copyArray( T *dst, T const*src, int length ) { // this might already be in standard C++ library?
        memcpy( dst, src, length * sizeof(T) );
    }
    static int getTotalNumWeights( NeuralNet *net ) {
        int totalWeightsSize = 0;
        for( int layerIdx = 1; layerIdx < net->layers.size(); layerIdx++ ) {
            Layer *layer = net->layers[layerIdx];
            totalWeightsSize += layer->getPersistSize();
        }
        return totalWeightsSize;
    }
    static void copyNetWeightsToArray( NeuralNet *net, float *target ) {
        int pos = 0;
        for( int layerIdx = 1; layerIdx < net->layers.size(); layerIdx++ ) {
            Layer *layer = net->layers[layerIdx];
            int persistSize = layer->getPersistSize();
            if( persistSize > 0 ) {
                layer->persistToArray( &(target[pos]) );
            }
            pos += persistSize;
        }
    }
    static void copyArrayToNetWeights( float const*source, NeuralNet *net ) {
        int pos = 0;
        for( int layerIdx = 1; layerIdx < net->layers.size(); layerIdx++ ) {
        Layer *layer = net->layers[layerIdx];
            int persistSize = layer->getPersistSize();
            if( persistSize > 0 ) {
                layer->unpersistFromArray( &(source[pos]) );
            }
            pos += persistSize;
        }
    }
    static void persistWeights( std::string filepath, std::string trainingConfigString, NeuralNet *net, int epoch, int batch, float annealedLearningRate, int numRight, float loss ) {
        int headerLength = 1024;
        int totalWeightsSize = getTotalNumWeights( net );
//        float *allWeightsArray = new float[totalWeightsSize];
        char *persistArray = new char[headerLength + totalWeightsSize * sizeof(float) ];
        int *persistArrayInts = reinterpret_cast<int *>(persistArray);
        float *persistArrayFloats = reinterpret_cast<float *>(persistArray);
        sprintf( persistArray, "ClCn" ); // so easy to recognise file type
        persistArrayInts[1] = 1; // data file version number
        persistArrayInts[2] = epoch;
        persistArrayInts[3] = batch;
        persistArrayInts[4] = numRight;
        persistArrayFloats[5] = loss;
        persistArrayFloats[6] = annealedLearningRate;
        sprintf( persistArray + 7 * 4, "%s", trainingConfigString.c_str() );
        copyNetWeightsToArray( net, reinterpret_cast<float *>(persistArray + headerLength) );
        FileHelper::writeBinary( "~" + filepath, reinterpret_cast<char *>(persistArray), 
            headerLength + totalWeightsSize * sizeof(float) );
        FileHelper::remove( filepath );
        FileHelper::rename( "~" + filepath, filepath );
        std::cout << "wrote weights to file, filesize " << ( ( headerLength + totalWeightsSize ) *sizeof(float)/1024) << "KB" << std::endl;
        delete[] persistArray;
    }
    static bool loadWeights( std::string filepath, std::string trainingConfigString, NeuralNet *net, int *p_epoch, int *p_batch, float *p_annealedLearningRate, int *p_numRight, float *p_loss ) {
        if( FileHelper::exists( filepath ) ){
            int headerSize = 1024;
            long fileSize;
            char * data = FileHelper::readBinary( filepath, &fileSize );
            float *allWeightsArray = reinterpret_cast<float *>(data + headerSize);
            std::cout << "read weights from file "  << (fileSize/1024) << "KB" << std::endl;
            int expectedTotalWeightsSize = getTotalNumWeights( net );
            int numFloatsRead = ( fileSize - headerSize ) / sizeof( float );
            if( expectedTotalWeightsSize != numFloatsRead ) {
                throw std::runtime_error("weights file contains " + toString(numFloatsRead) + " floats, but we expect to see: " + toString( expectedTotalWeightsSize ) + ".  So there is probably some mismatch between the weights file, and the settings, or network version, used." );
            }
            int *dataAsInts = reinterpret_cast<int *>(data);
            float *dataAsFloats = reinterpret_cast<float *>(data);
            if( data[0] != 'C' || data[1] != 'l' || data[2] != 'C' || data[3] != 'n' ) {
                std::cout << "weights file not ClConvolve format" << std::endl;
                return false;
            }
            if( dataAsInts[1] != 1 ) {
                std::cout << "weights file version not known" << std::endl;
                return false;
            }
            if( trainingConfigString != std::string( data + 7 * 4 ) ) {
                std::cout << "training options dont match weights file" << std::endl;
                return false;
            }
            *p_epoch = dataAsInts[2];
            *p_batch = dataAsInts[3];
            *p_numRight = dataAsInts[4];
            *p_loss = dataAsFloats[5];
            *p_annealedLearningRate = dataAsFloats[6];
            copyArrayToNetWeights( allWeightsArray, net );
            delete [] data;
            return true;
        }
        return false;
    }
};

