#include "Layer.h"

using namespace std;

Layer::Layer( Layer *previousLayer, LayerMaker const*maker ) :
    previousLayer( previousLayer ),
    numPlanes( maker->getNumPlanes() ),
    boardSize( maker->getBoardSize() ),
    boardSizeSquared( boardSize * boardSize ),
    results(0),
    weights(0),
    biasWeights(0),
    biased(maker->_biased),
    activationFunction( maker->_activationFunction ),
    upstreamBoardSize( previousLayer == 0 ? 0 : previousLayer->boardSize ),
    upstreamNumPlanes( previousLayer == 0 ? 0 : previousLayer->numPlanes ),
    upstreamBoardSizeSquared( upstreamBoardSize * upstreamBoardSize ),
    layerIndex( previousLayer == 0 ? 0 : previousLayer->layerIndex + 1 ),
    weOwnResults(false) {
}

Layer::~Layer() { // [virtual]
    if( results != 0 && weOwnResults ) {
         delete[] results;
    }
    if( weights != 0 ) {
        delete[] weights;
    }
    if( biasWeights != 0 ) {
        delete[] biasWeights;
    }
    if( activationFunction != 0 ) {
        delete activationFunction;
    }
}
//    inline float activationFn( float value ) {
//        //return 1.7159 * tanh( value );
//        return tanh( value );
//    }
// used to set up internal buffers and stuff
void Layer::setBatchSize( int batchSize ) {  // [virtual]
    throw std::runtime_error("setBatchsize not implemetned for this layer type");
}
bool Layer::hasResultsWrapper() const { // [virtual]
    return false;
}
CLWrapper *Layer::getResultsWrapper() { // [virtual]
    throw std::runtime_error("getResultsWrapper not implemetned for this layer type, layer " + toString(layerIndex) );
}
float * Layer::getResults() { // [virtual]
    return results;
};
int Layer::getResultsSize() const { // [virtual]
//        throw std::runtime_error("getResultsSize not implemented for this layer type");
     return numPlanes * boardSize * boardSize * batchSize;
}
int Layer::getNumPlanes() const {
    return numPlanes;
}
int Layer::getBoardSize() const {
    return boardSize;
}
void Layer::propagate() { // [virtual]
    throw std::runtime_error("propagate not implemented for this layer type");
}
void Layer::print() const {  // [virtual]
//        std::cout << "print() not implemented for this layer type" << std:: endl; 
    printWeights();
    if( results != 0 ) {
        printOutput();
    } else {
        std::cout << "No results yet " << std::endl;
    }
}
void Layer::initWeights( float*weights ) { // [virtual]
    int numWeights = getWeightsSize();
    for( int i = 0; i < numWeights; i++ ) {
        this->weights[i] = weights[i];
    }
}
void Layer::initBiasWeights( float*biasWeights ) { // [virtual]
    int numBiasWeights = getBiasWeightsSize();
    for( int i = 0; i < numBiasWeights; i++ ) {
        this->biasWeights[i] = biasWeights[i];
    }
}
void Layer::printWeightsAsCode() const { // [virtual]
    std::cout << "float weights" << layerIndex << "[] = {";
    const int numWeights = getWeightsSize();
    for( int i = 0; i < numWeights; i++ ) {
        std::cout << weights[i];
        if( i < numWeights - 1 ) std::cout << ", ";
        if( i > 0 && i % 20 == 0 ) std::cout << std::endl;
    }
    std::cout << "};" << std::endl;
//        std::cout << netObjectName << "->layers[" << layerIndex << "]->weights[
}
void Layer::printBiasWeightsAsCode() const { // [virtual]
    std::cout << "float biasWeights" << layerIndex << "[] = {";
    const int numBiasWeights = getBiasWeightsSize();
    for( int i = 0; i < numBiasWeights; i++ ) {
        std::cout << biasWeights[i];
        if( i < numBiasWeights - 1 ) std::cout << ", ";
        if( i > 0 && i % 20 == 0 ) std::cout << std::endl;
    }
    std::cout << "};" << std::endl;
//        std::cout << netObjectName << "->layers[" << layerIndex << "]->weights[
}
void Layer::printWeights() const {  // [virtual]
    std::cout << "printWeights() not implemented for this layer type" << std:: endl; 
}
void Layer::printOutput() const {  // [virtual]
    std::cout << "printOutpu() not implemented for this layer type" << std:: endl; 
}
void Layer::backPropExpected( float learningRate, float const *expected ) { // [virtual]
    throw std::runtime_error("backPropExpected not implemented for this layertype, layerindex " + toString(layerIndex ) );
}
void Layer::backPropErrors( float learningRate, float const *errors, float *errorsForUpstream ) { // [virtual]
    throw std::runtime_error("backproperrors not implemented for this layertype, layerindex " + toString(layerIndex ) );
}
int Layer::getWeightsSize() const { // [virtual]
    throw std::runtime_error("getWeightsSize not implemented for this layertype");
}
int Layer::getBiasWeightsSize() const { // [virtual]
    throw std::runtime_error("getBiasWeightsSize not implemented for this layertype");
}
void Layer::calcErrors( float const *expected, float *errors ) { // [virtual]
    throw std::runtime_error("calcErrors not implemented for this layertype, layerindex " + toString(layerIndex ) );
}
float Layer::calcLoss( float const *expected ) {
    float E = 0;
    getResults();
    // this is matrix subtraction, then element-wise square, then aggregation
    for( int imageId = 0; imageId < batchSize; imageId++ ) {
        for( int plane = 0; plane < numPlanes; plane++ ) {
            for( int outRow = 0; outRow < boardSize; outRow++ ) {
                for( int outCol = 0; outCol < boardSize; outCol++ ) {
                    int resultOffset = getResultIndex( imageId, plane, outRow, outCol ); //imageId * numPlanes + out;
                    float expectedOutput = expected[resultOffset];
                    float actualOutput = results[resultOffset];
                    float diff = expectedOutput - actualOutput;
                    float squarederror = diff * diff;
                    E += squarederror;
//                        std::cout << " image " << imageId << " outplane " << plane << " i " << outRow << " j " << outCol <<
//                           " expected " << expectedOutput << " actual " << actualOutput << " squarederror " << squarederror
//                            << std::endl;
                }
            }
        }            
    }
    return E;
 }

