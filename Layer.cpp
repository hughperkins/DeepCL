#include "Layer.h"

using namespace std;

Layer::Layer( Layer *previousLayer, LayerMaker const*maker ) :
    previousLayer( previousLayer ),
    nextLayer( 0 ),
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
    if( previousLayer != 0 ) {
        previousLayer->nextLayer = this;
    }
}
Layer::Layer( Layer *previousLayer, ExpectedValuesLayerMaker const*maker ) :
        previousLayer( previousLayer ),
        nextLayer(0),
        numPlanes( previousLayer->numPlanes ),
        boardSize( previousLayer->boardSize ),
        boardSizeSquared( previousLayer->boardSizeSquared ),
        results(0),
        weights(0),
        biasWeights(0),
        biased(false),
        activationFunction(0),
        upstreamBoardSize(previousLayer->boardSize),
        upstreamNumPlanes( previousLayer->numPlanes),
        upstreamBoardSizeSquared( previousLayer->boardSizeSquared ),
        layerIndex( previousLayer->layerIndex + 1 ),
        weOwnResults(false) {
    if( previousLayer != 0 ) {
        previousLayer->nextLayer = this;
        this->results = previousLayer->results;
    }
}
// [virtual]
Layer::~Layer() {
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
// [virtual]
void Layer::setBatchSize( int batchSize ) {
    throw std::runtime_error("setBatchsize not implemetned for this layer type");
}
// [virtual]
bool Layer::providesErrorsWrapper() const {
    return false;
}
// [virtual]
float *Layer::getErrorsForUpstream() {
    throw std::runtime_error("getErrorsForUpstream not implemented for this layer type, layer " + toString(layerIndex) );
}
// [virtual]
CLWrapper *Layer::getErrorsForUpstreamWrapper() {
    throw std::runtime_error("getErrorsForUpstreamWrapper not implemented for this layer type, layer " + toString(layerIndex) );
}
// [virtual]
bool Layer::hasResultsWrapper() const {
    return false;
}
// [virtual]
CLWrapper *Layer::getResultsWrapper() {
    throw std::runtime_error("getResultsWrapper not implemetned for this layer type, layer " + toString(layerIndex) );
}
// [virtual]
int Layer::getResultsSize() const {
//        throw std::runtime_error("getResultsSize not implemented for this layer type");
     return numPlanes * boardSize * boardSize * batchSize;
}
int Layer::getNumPlanes() const {
    return numPlanes;
}
int Layer::getBoardSize() const {
    return boardSize;
}
// [virtual]
void Layer::propagate() {
    throw std::runtime_error("propagate not implemented for this layer type");
}
// [virtual]
void Layer::print() const {
//        std::cout << "print() not implemented for this layer type" << std:: endl; 
    printWeights();
    if( results != 0 ) {
        printOutput();
    } else {
        std::cout << "No results yet " << std::endl;
    }
}
// [virtual]
void Layer::initWeights( float*weights ) {
    int numWeights = getWeightsSize();
    for( int i = 0; i < numWeights; i++ ) {
        this->weights[i] = weights[i];
    }
}
// [virtual]
void Layer::initBiasWeights( float*biasWeights ) {
    int numBiasWeights = getBiasWeightsSize();
    for( int i = 0; i < numBiasWeights; i++ ) {
        this->biasWeights[i] = biasWeights[i];
    }
}
// [virtual]
void Layer::printWeightsAsCode() const {
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
// [virtual]
void Layer::printBiasWeightsAsCode() const {
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
// [virtual]
void Layer::printWeights() const {
    std::cout << "printWeights() not implemented for this layer type" << std:: endl; 
}
// [virtual]
void Layer::printOutput() const {
    std::cout << "printOutpu() not implemented for this layer type" << std:: endl; 
}
//// [virtual]
//void Layer::backPropExpected( float learningRate, float const *expected ) {
//    throw std::runtime_error("backPropExpected not implemented for this layertype, layerindex " + toString(layerIndex ) );
//}
// [virtual]
void Layer::backPropErrors( float learningRate ) {
    throw std::runtime_error("backproperrors not implemented for this layertype, layerindex " + toString(layerIndex ) );
}
// [virtual]
int Layer::getWeightsSize() const {
    throw std::runtime_error("getWeightsSize not implemented for this layertype");
}
// [virtual]
int Layer::getBiasWeightsSize() const {
    throw std::runtime_error("getBiasWeightsSize not implemented for this layertype");
}
//// [virtual]
//void Layer::calcErrors( float const *expected, float *errors ) {
//    throw std::runtime_error("calcErrors not implemented for this layertype, layerindex " + toString(layerIndex ) );
//}
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

