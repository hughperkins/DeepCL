#include "layer/Layer.h"
#include "weights/WeightsPersister.h"
#include "CppRuntimeBoundary.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

PUBLICAPI Layer::Layer(Layer *previousLayer, LayerMaker2 *maker) :
    previousLayer(previousLayer),
    nextLayer(0),
    layerIndex(previousLayer == 0 ? 0 : previousLayer->layerIndex + 1),
    training(false),
    maker(maker)
     {
    if(previousLayer != 0) {
        previousLayer->nextLayer = this;
    }
}
VIRTUAL Layer::~Layer() {
    if(maker != 0) {
        //delete maker; // this segfaults sometimes, (probably because it already
                        // self-deleted)
    }
}
/// \brief Are we training or predicting?
/// Only affects the Random translations and patches layers currently
PUBLICAPI VIRTUAL void Layer::setTraining(bool training) {
    this->training = training;
}
/// used to set up internal buffers and stuff
PUBLICAPI VIRTUAL void Layer::setBatchSize(int batchSize) {
    throw std::runtime_error("setBatchsize not implemetned for this layer type");
}
VIRTUAL bool Layer::providesGradInputWrapper() const {
    return false;
}
VIRTUAL const char *Layer::getClassNameAsCharStar() const {
    return deepcl_stringToCharStar(getClassName());
}
VIRTUAL float *Layer::getGradInput() {
    throw std::runtime_error("getGradInput not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getGradWeightsWrapper() {
    throw std::runtime_error("getGradWeightsWrapper not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getGradBiasWrapper() {
    throw std::runtime_error("getGradBiasWrapper not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getWeightsWrapper() {
    throw std::runtime_error("getWeightsWrapper not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getBiasWrapper() {
    throw std::runtime_error("getBiasWrapper not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getGradInputWrapper() {
    throw std::runtime_error("getGradInputWrapper not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL bool Layer::getBiased() const {
     throw std::runtime_error("getBiased not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL bool Layer::hasOutputWrapper() const {
    return false;
}
PUBLICAPI VIRTUAL CLWrapper *Layer::getOutputWrapper() {
    throw std::runtime_error("getOutputWrapper not implemetned for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getOutputCubeSize() const {
    throw std::runtime_error("getOutputCubeSize not implemetned for " + getClassName());
 //     return numPlanes * imageSize * imageSize * batchSize;
}
PUBLICAPI VIRTUAL int Layer::getOutputPlanes() const {
    throw std::runtime_error("getOutputPlanes not implemetned for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getOutputSize() const {
    throw std::runtime_error("getOutputSize not implemetned for " + getClassName());
}
VIRTUAL void Layer::forward() {
    throw std::runtime_error("forward not implemented for " + getClassName());
}
VIRTUAL bool Layer::needsBackProp() {
    throw std::runtime_error("needsBackProp not implemented for " + getClassName());
}
VIRTUAL void Layer::print() {
//    printWeights();
//    if(output != 0) {
    printOutput();
    printWeights();
//    } else {
//        std::cout << "No output yet " << std::endl;
//    }
}
VIRTUAL void Layer::initWeights(float const*weights) {
    throw std::runtime_error("initWeights not implemetned for " + getClassName());
//    int numWeights = getWeightsSize();
//    for(int i = 0; i < numWeights; i++) {
//        this->weights[i] = weights[i];
//    }
}
VIRTUAL void Layer::initBias(float const *bias) {
    throw std::runtime_error("initBias not implemetned for " + getClassName());
//    int numBias = getBiasSize();
//    for(int i = 0; i < numBias; i++) {
//        this->bias[i] = bias[i];
//    }
}
int Layer::getLayerIndex() {
    return layerIndex;
}
VIRTUAL void Layer::printWeights() {
    throw std::runtime_error("printWeights not implemented for " + getClassName());
}
VIRTUAL void Layer::printOutput() {
    throw std::runtime_error("printOutput not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL void Layer::backward() {
    throw std::runtime_error("backward not implemented for " + getClassName());
}
VIRTUAL float *Layer::getGradWeights() {
    throw std::runtime_error("getGradWeights not implemented for " + getClassName());
}
VIRTUAL float *Layer::getGradBias() {
    throw std::runtime_error("getGradBias not implemented for " + getClassName());
}
VIRTUAL bool Layer::biased() {
    throw std::runtime_error("biased not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getWeightsSize() const {
    throw std::runtime_error("getWeightsSize not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getBiasSize() const {
    throw std::runtime_error("getBiasSize not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getPersistSize() const {
    return getPersistSize(WeightsPersister::latestVersion);
}
PUBLICAPI VIRTUAL void Layer::persistToArray(float *array) {
    persistToArray(WeightsPersister::latestVersion, array);
}
/// \brief store the current weights and biases to array
/// Note that you need to allocate array first
PUBLICAPI VIRTUAL void Layer::persistToArray(int version, float *array) {
    throw std::runtime_error("persistToArray not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL void Layer::unpersistFromArray(float const*array) {
    unpersistFromArray(WeightsPersister::latestVersion, array);
}
/// \brief initialize the current weights and biases from array
PUBLICAPI VIRTUAL void Layer::unpersistFromArray(int version, float const*array) {
    throw std::runtime_error("unpersistFromArray not implemented for " + getClassName());
}
VIRTUAL void Layer::setWeights(float *weights, float *bias) {
    throw std::runtime_error("setWeights not implemented for " + getClassName());
}
VIRTUAL float const *Layer::getWeights() const {
    throw std::runtime_error("getWeights const not implemented for " + getClassName());
}
VIRTUAL float *Layer::getWeights() {
    throw std::runtime_error("getWeights not implemented for " + getClassName());
}
VIRTUAL float *Layer::getBias() {
    throw std::runtime_error("getBias not implemented for " + getClassName());
}
VIRTUAL float const*Layer::getBias() const {
    throw std::runtime_error("getBias const not implemented for " + getClassName());
}
/// \brief Get a string representation of the layer
VIRTUAL std::string Layer::asString() const {
    return "Layer{}";
}
VIRTUAL const char *Layer::asNewCharStar() const {
    return deepcl_stringToCharStar(asString());
}
VIRTUAL bool Layer::needsTrainerState  () const {
    return false;
}
// This transfers ownership of the trainer to the layer,
// which is responsible for deleting it
// probably should pass in a Maker class instead
VIRTUAL void Layer::setTrainerState(TrainerStateMaker *trainerMaker) {
    throw std::runtime_error("setTrainer not implemented for " + getClassName());
}
VIRTUAL TrainerState *Layer::getTrainerState() {
    throw std::runtime_error("getTrainerState not implemented for " + getClassName());
}
VIRTUAL TrainerState *Layer::getBiasTrainerState() {
    throw std::runtime_error("getBiasTrainerState not implemented for " + getClassName());
}
VIRTUAL void Layer::updateWeights(CLWrapper *weightChangesWrapper, CLWrapper *biasChangesWrapper) {
    throw std::runtime_error("updateWeights not implemented for " + getClassName());
}

