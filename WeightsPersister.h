#pragma once

#include <iostream>

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
            totalWeightsSize += layer->getWeightsSize();
            totalWeightsSize += layer->getBiasWeightsSize();
        }
        return totalWeightsSize;
    }
    static void persistWeights( std::string filepath, NeuralNet *net ) {
        int totalWeightsSize = getTotalNumWeights( net );
        float *allWeightsArray = new float[totalWeightsSize];
        int pos = 0;
        for( int layerIdx = 1; layerIdx < net->layers.size(); layerIdx++ ) {
            Layer *layer = net->layers[layerIdx];
            copyArray( &(allWeightsArray[pos]), layer->weights, layer->getWeightsSize() );
            pos += layer->getWeightsSize();
            copyArray( &(allWeightsArray[pos]), layer->biasWeights, layer->getBiasWeightsSize() );
            pos += layer->getBiasWeightsSize();
        }
        FileHelper::writeBinary( "~" + filepath, reinterpret_cast<char *>(allWeightsArray), 
            totalWeightsSize * sizeof(float) );
        FileHelper::remove( filepath );
        FileHelper::rename( "~" + filepath, filepath );
        cout << "wrote weights to file, size " << (totalWeightsSize*sizeof(float)/1024) << "KB" << endl;
        delete[] allWeightsArray;
    }
    static void loadWeights( std::string filepath, NeuralNet *net ) {
        if( FileHelper::exists( filepath ) ){
            long fileSize;
            char * data = FileHelper::readBinary( filepath, &fileSize );
            float *allWeightsArray = reinterpret_cast<float *>(data);
            cout << "read weights from file "  << (fileSize/1024) << "KB" << endl;
            int expectedTotalWeightsSize = getTotalNumWeights( net );
            int numFloatsRead = fileSize / sizeof( float );
            if( expectedTotalWeightsSize != numFloatsRead ) {
                throw std::runtime_error("weights file contains " + toString(numFloatsRead) + " floats, but we expect to see: " + toString( expectedTotalWeightsSize ) + ".  So there is probably some mismatch between the weights file, and the settings, or network version, used." );
            }
            int pos = 0;
            for( int layerIdx = 1; layerIdx < net->layers.size(); layerIdx++ ) {
            Layer *layer = net->layers[layerIdx];
                copyArray( layer->weights, &(allWeightsArray[pos]), layer->getWeightsSize() );
                pos += layer->getWeightsSize();
                copyArray( layer->biasWeights, &(allWeightsArray[pos]), layer->getBiasWeightsSize() );
                pos += layer->getBiasWeightsSize();
            }
            delete [] data;
        }
    }
};

