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
    void copyArray( T *dst, T const*src, length ) { // this might already be in standard C++ library?
        memcpy( dst, src, length ( sizeof(T) );
    }
    void persistWeights( std::string filepath, NeuralNet *net ) {
        int totalWeightsSize = 0;
        for( int layer = 1; layer < net->layers.size(); layer++ ) {
            totalWeightsSize += layer->getWeightsSize();
            totalWeightsSize += layer->getBiasWeightsSize();
        }
        float *allWeightsArray = new float[totalWeightsSize];
        int pos = 0;
        for( int layer = 1; layer < net->layers.size(); layer++ ) {
            copyArray( &(allWeightsArray[pos]), layer->weights, layer->getWeightsSize() );
            pos += layer->getWeightsSize();
            copyArray( &(allWeightsArray[pos]), layer->biasWeights, layer->getBiasWeightsSize() );
            pos += layer->getBiasWeightsSize();
        }
        FileHelper::writeBinary( filepath, reinterpret_cast<unsigned char *>(allWeightsArray), 
            totalWeightsSize * sizeof(float) );
        cout << "wrote weights to file, size " << (totalWeightsSize/1024) << "KB" << endl;
        delete[] allWeightsArray;
    }
    void loadWeights( std::string filepath, NeuralNet *net ) {
        if( FileHelper::exists( filepath ) ){
            int fileSize;
            unsigned char * data = FileHelper::readBinary( filepath, &fileSize );
            float *allWeightsArray = reinterpret_cast<float *>(data);
            cout << "read data from file "  << (fileSize/1024) << "KB" << endl;
            int pos = 0;
            for( int layer = 1; layer < net->layers.size(); layer++ ) {
                copyArray( layer->weights, &(allWeightsArray[pos]), layer->getWeightsSize() );
                pos += layer->getWeightsSize();
                copyArray( layer->biasWeights, &(allWeightsArray[pos]), layer->getBiasWeightsSize() );
                pos += layer->getBiasWeightsSize();
            }
            delete [] data;
        }
    }
};

