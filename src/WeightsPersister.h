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
//            totalWeightsSize += layer->getBiasWeightsSize();
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
//            copyArray( &(target[pos]), layer->getWeights(), layer->getWeightsSize() );
//            pos += layer->getWeightsSize();
//            copyArray( &(target[pos]), layer->getBiasWeights(), layer->getBiasWeightsSize() );
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
//            layer->initWeights( &(source[pos]) );
//            pos += layer->getWeightsSize();
//            layer->initBiasWeights( &(source[pos]) );
//            pos += layer->getBiasWeightsSize();
            pos += persistSize;
        }
    }
    static void persistWeights( std::string filepath, NeuralNet *net ) {
        int totalWeightsSize = getTotalNumWeights( net );
        float *allWeightsArray = new float[totalWeightsSize];
        copyNetWeightsToArray( net, allWeightsArray );
//        int pos = 0;
//        for( int layerIdx = 1; layerIdx < net->layers.size(); layerIdx++ ) {
//            Layer *layer = net->layers[layerIdx];
//            copyArray( &(allWeightsArray[pos]), layer->getWeights(), layer->getWeightsSize() );
//            pos += layer->getWeightsSize();
//            copyArray( &(allWeightsArray[pos]), layer->getBiasWeights(), layer->getBiasWeightsSize() );
//            pos += layer->getBiasWeightsSize();
//        }
        FileHelper::writeBinary( "~" + filepath, reinterpret_cast<char *>(allWeightsArray), 
            totalWeightsSize * sizeof(float) );
        FileHelper::remove( filepath );
        FileHelper::rename( "~" + filepath, filepath );
        std::cout << "wrote weights to file, size " << (totalWeightsSize*sizeof(float)/1024) << "KB" << std::endl;
        delete[] allWeightsArray;
    }
    static void loadWeights( std::string filepath, NeuralNet *net ) {
        if( FileHelper::exists( filepath ) ){
            long fileSize;
            char * data = FileHelper::readBinary( filepath, &fileSize );
            float *allWeightsArray = reinterpret_cast<float *>(data);
            std::cout << "read weights from file "  << (fileSize/1024) << "KB" << std::endl;
            int expectedTotalWeightsSize = getTotalNumWeights( net );
            int numFloatsRead = fileSize / sizeof( float );
            if( expectedTotalWeightsSize != numFloatsRead ) {
                throw std::runtime_error("weights file contains " + toString(numFloatsRead) + " floats, but we expect to see: " + toString( expectedTotalWeightsSize ) + ".  So there is probably some mismatch between the weights file, and the settings, or network version, used." );
            }
            copyArrayToNetWeights( allWeightsArray, net );
//            int pos = 0;
//            for( int layerIdx = 1; layerIdx < net->layers.size(); layerIdx++ ) {
//            Layer *layer = net->layers[layerIdx];
//                layer->initWeights( &(allWeightsArray[pos]) );
//                pos += layer->getWeightsSize();
//                layer->initBiasWeights( &(allWeightsArray[pos]) );
//                pos += layer->getBiasWeightsSize();
//            }
            delete [] data;
        }
    }
};

