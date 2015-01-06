#include <iostream>

class WeightsPersister {
public:
    void persistWeights( std::string filepath, NeuralNet *net ) {
        int totalWeights = 0;
        for( int layer = 1; layer < net->layers.size(); layer++ ) {
            totalWeights += layer->getWeightsSize();
            totalWeights += layer->getBiasWeightsSize();
        }
        FileHelper::writeBinary( "weights.dat", reinterpret_cast<unsigned char *>(net->layers[1]->weights), 
            net->layers[1]->getWeightsSize() * sizeof(float) );
        cout << "wrote weights to file " << endl;
        FileHelper::writeBinary( "biasweights.dat", reinterpret_cast<unsigned char *>(dynamic_cast<ConvolutionalLayer*>(net->layers[1])->biasWeights), 
            dynamic_cast<ConvolutionalLayer*>(net->layers[1])->getBiasWeightsSize() * sizeof(float) );
    }
    void loadWeights( std::string filepath, NeuralNet *net ) {
    }
};

