#include "Layer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

Layer::Layer( Layer *previousLayer, LayerMaker const*maker ) :
    previousLayer( previousLayer ),
    nextLayer( 0 ),
//    numPlanes( maker->getNumPlanes() ),
//    boardSize( maker->getBoardSize() ),
//    boardSizeSquared( boardSize * boardSize ),
//    results(0),
//    weights(0),
//    biasWeights(0),
//    biased(maker->getBiased()),
//    activationFunction( maker->getActivationFunction() ),
//    upstreamBoardSize( previousLayer == 0 ? 0 : previousLayer->boardSize ),
//    upstreamNumPlanes( previousLayer == 0 ? 0 : previousLayer->numPlanes ),
//    upstreamBoardSizeSquared( upstreamBoardSize * upstreamBoardSize ),
    layerIndex( previousLayer == 0 ? 0 : previousLayer->layerIndex + 1 )
//    weOwnResults(false)
     {
    if( previousLayer != 0 ) {
        previousLayer->nextLayer = this;
    }
}
//Layer::Layer( Layer *previousLayer, ExpectedValuesLayerMaker const*maker ) :
//        previousLayer( previousLayer ),
//        nextLayer(0),
//        numPlanes( previousLayer->numPlanes ),
//        boardSize( previousLayer->boardSize ),
//        boardSizeSquared( previousLayer->boardSizeSquared ),
//        results(0),
//        weights(0),
//        biasWeights(0),
//        biased(false),
////        activationFunction(0),
//        upstreamBoardSize(previousLayer->boardSize),
//        upstreamNumPlanes( previousLayer->numPlanes),
//        upstreamBoardSizeSquared( previousLayer->boardSizeSquared ),
//        layerIndex( previousLayer->layerIndex + 1 ),
//        weOwnResults(false) {
//    if( previousLayer != 0 ) {
//        previousLayer->nextLayer = this;
//        this->results = previousLayer->results;
//    }
//}
VIRTUAL Layer::~Layer() {
//    if( results != 0 && weOwnResults ) {
//         delete[] results;
//    }
//    if( weights != 0 ) {
//        delete[] weights;
//    }
//    if( biasWeights != 0 ) {
//        delete[] biasWeights;
//    }
//    if( activationFunction != 0 ) {
//        delete activationFunction;
//    }
}
//    inline float activationFn( float value ) {
//        //return 1.7159 * tanh( value );
//        return tanh( value );
//    }
// used to set up internal buffers and stuff
VIRTUAL void Layer::setBatchSize( int batchSize ) {
    throw std::runtime_error("setBatchsize not implemetned for this layer type");
}
VIRTUAL bool Layer::providesErrorsForUpstreamWrapper() const {
    return false;
}
VIRTUAL float *Layer::getErrorsForUpstream() {
    throw std::runtime_error("getErrorsForUpstream not implemented for this layer type, layer " + toString(layerIndex) );
}
VIRTUAL CLWrapper *Layer::getErrorsForUpstreamWrapper() {
    throw std::runtime_error("getErrorsForUpstreamWrapper not implemented for this layer type, layer " + toString(layerIndex) );
}
VIRTUAL bool Layer::getBiased() const {
     throw std::runtime_error("getBiased not implemented for this layer type, layer " + toString(layerIndex) );
}
VIRTUAL bool Layer::hasResultsWrapper() const {
    return false;
}
VIRTUAL CLWrapper *Layer::getResultsWrapper() {
    throw std::runtime_error("getResultsWrapper not implemetned for this layer type, layer " + toString(layerIndex) );
}
VIRTUAL ActivationFunction const*Layer::getActivationFunction() {
    throw std::runtime_error("getActivationFunction not implemetned for this layer type, layer " + toString(layerIndex) );
}
VIRTUAL int Layer::getResultsSize() const {
    throw std::runtime_error("getResultsSize not implemetned for this layer type, layer " + toString(layerIndex) + " " + toString(this) );
 //     return numPlanes * boardSize * boardSize * batchSize;
}
VIRTUAL int Layer::getOutputCubeSize() const {
    throw std::runtime_error("getOutputCubeSize not implemetned for this layer type, layer " + toString(layerIndex) + " " + toString(this) );
 //     return numPlanes * boardSize * boardSize * batchSize;
}
VIRTUAL int Layer::getOutputPlanes() const {
    throw std::runtime_error("getOutputPlanes not implemetned for this layer type, layer " + toString(layerIndex) + " " + toString(this) );
}
VIRTUAL int Layer::getOutputBoardSize() const {
    throw std::runtime_error("getOutputBoardSize not implemetned for this layer type, layer " + toString(layerIndex) + " " + toString(this) );
}
VIRTUAL void Layer::propagate() {
    throw std::runtime_error("propagate not implemented for this layer type");
}
VIRTUAL void Layer::print() {
//    printWeights();
//    if( results != 0 ) {
    printOutput();
    printWeights();
//    } else {
//        std::cout << "No results yet " << std::endl;
//    }
}
VIRTUAL void Layer::initWeights( float const*weights ) {
    throw std::runtime_error("initWeights not implemetned for this layer type, layer " + toString(layerIndex) + " " + toString(this) );
//    int numWeights = getWeightsSize();
//    for( int i = 0; i < numWeights; i++ ) {
//        this->weights[i] = weights[i];
//    }
}
VIRTUAL void Layer::initBiasWeights( float const *biasWeights ) {
    throw std::runtime_error("initBiasWeights not implemetned for this layer type, layer " + toString(layerIndex) + " " + toString(this) );
//    int numBiasWeights = getBiasWeightsSize();
//    for( int i = 0; i < numBiasWeights; i++ ) {
//        this->biasWeights[i] = biasWeights[i];
//    }
}
VIRTUAL void Layer::printWeightsAsCode() const {
    std::cout << "float weights" << layerIndex << "[] = {";
    const int numWeights = getWeightsSize();
    float const*weights = getWeights();
    for( int i = 0; i < numWeights; i++ ) {
        std::cout << weights[i];
        if( i < numWeights - 1 ) std::cout << ", ";
        if( i > 0 && i % 20 == 0 ) std::cout << std::endl;
    }
    std::cout << "};" << std::endl;
//        std::cout << netObjectName << "->layers[" << layerIndex << "]->weights[
}
VIRTUAL void Layer::printBiasWeightsAsCode() const {
    std::cout << "float biasWeights" << layerIndex << "[] = {";
    const int numBiasWeights = getBiasWeightsSize();
    float const*biasWeights = getBiasWeights();
    for( int i = 0; i < numBiasWeights; i++ ) {
        std::cout << biasWeights[i];
        if( i < numBiasWeights - 1 ) std::cout << ", ";
        if( i > 0 && i % 20 == 0 ) std::cout << std::endl;
    }
    std::cout << "};" << std::endl;
//        std::cout << netObjectName << "->layers[" << layerIndex << "]->weights[
}
VIRTUAL void Layer::printWeights() {
    std::cout << "printWeights() not implemented for this layer type" << std:: endl; 
}
VIRTUAL void Layer::printOutput() const {
    std::cout << "printOutpu() not implemented for this layer type" << std:: endl; 
}
VIRTUAL void Layer::backProp( float learningRate ) {
    throw std::runtime_error("backProp not implemented for this layertype, layerindex " + toString(layerIndex ) );
}
VIRTUAL int Layer::getWeightsSize() const {
    throw std::runtime_error("getWeightsSize not implemented for this layertype");
}
VIRTUAL int Layer::getBiasWeightsSize() const {
    throw std::runtime_error("getBiasWeightsSize not implemented for this layertype");
}
VIRTUAL int Layer::getPersistSize() const {
    throw std::runtime_error("getPersistSize not implemented for this layertype, layerindex " + toString(layerIndex ) );
}
VIRTUAL void Layer::persistToArray(float *array) {
    throw std::runtime_error("persistToArray not implemented for this layertype, layerindex " + toString(layerIndex ) );
}
VIRTUAL void Layer::unpersistFromArray(float const*array) {
    throw std::runtime_error("unpersistFromArray not implemented for this layertype, layerindex " + toString(layerIndex ) );
}
VIRTUAL void Layer::setWeights(float *weights, float *biasWeights) {
    throw std::runtime_error("setWeights not implemented for this layertype");
}
VIRTUAL float const *Layer::getWeights() const {
    throw std::runtime_error("getWeights const not implemented for this layertype");
}
VIRTUAL float *Layer::getWeights() {
    throw std::runtime_error("getWeights not implemented for this layertype " + toString(layerIndex) );
}
VIRTUAL float const*Layer::getBiasWeights() const {
    throw std::runtime_error("getBiasWeights not implemented for this layertype");
}
VIRTUAL std::string Layer::asString() const {
    return "Layer{}";
}

ostream &operator<<(ostream&os, Layer const*layer ) {
    os << layer->asString();
    return os;
}

