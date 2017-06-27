// Copyright Hugh Perkins 2014 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "util/StatefulTimer.h"

#include "layer/LayerMaker.h"
#include "loss/SoftMaxLayer.h"
#include <cfloat>

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

SoftMaxLayer::SoftMaxLayer(Layer *previousLayer, SoftMaxMaker *maker) :
    LossLayer(previousLayer, maker),
        perPlane(maker->_perPlane),
        imageSize(previousLayer->getOutputSize()),
        numPlanes(previousLayer->getOutputPlanes()),
        imageSizeSquared(previousLayer->getOutputSize() * previousLayer->getOutputSize()),
        output(0),
        gradInput(0),
        allocatedSize(0),
        batchSize(0)
         {
}
VIRTUAL SoftMaxLayer::~SoftMaxLayer() {
    if(gradInput != 0) {
        delete[] gradInput;
    }
    if(output != 0) {
        delete[] output;
    }
}
VIRTUAL std::string SoftMaxLayer::getClassName() const {
    return "SoftMaxLayer";
}
VIRTUAL float *SoftMaxLayer::getOutput() {
    return output;
}
VIRTUAL float *SoftMaxLayer::getGradInput() {
    return gradInput;
}
VIRTUAL void SoftMaxLayer::setBatchSize(int batchSize) {
    this->batchSize = batchSize;
    if(batchSize <= this->allocatedSize) {
        return;
    }
    if(output != 0) {
        delete[] output;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    output = new float[ getOutputNumElements() ];
    gradInput = new float[ previousLayer-> getOutputNumElements() ];
    allocatedSize = batchSize;
}
VIRTUAL int SoftMaxLayer::getBatchSize() {
    return this->batchSize;
}
// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayer::calcLossFromLabels(int const *labels) {
//    cout << "softmaxlayer::calcloss" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcLossfromlabels");
    float loss = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int label = labels[n * numPlanes + plane];
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                loss += - log(std::max(output[ imageOffset + label ], FLT_MIN));
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            int label = labels[n];
            loss += - log(std::max(output[imageOffset + label], FLT_MIN));
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcLossfromlabels");
    return loss;
}
// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayer::calcLoss(float const *expectedValues) {
    StatefulTimer::timeCheck("start SoftMaxLayer calcLoss");
    float loss = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                for(int i = 0; i < imageSizeSquared; i++) {
                    if(expectedValues[ imageOffset + i ] != 0) {
                        float thisloss = - expectedValues[ imageOffset + i ] * log(std::max(output[ imageOffset + i ], FLT_MIN));
                        loss += thisloss;
                    }
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            for(int plane = 0; plane < numPlanes; plane++) {
                float thisloss = - expectedValues[imageOffset + plane] * log(std::max(output[imageOffset + plane], FLT_MIN));
                loss += thisloss;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcLoss");
    return loss;
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcGradInputFromLabels(int const *labels) {
//    cout << "softmaxlayer::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcGradInputfromlabels");
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                int label = labels[n * numPlanes + plane];
                for(int i = 0; i < imageSizeSquared; i++) {
                    float value = output[imageOffset + i];
                    if (std::isfinite(value) == false)
                        throw runtime_error("Output is a non-finite number, this usually means the learning rate is too high");
                    gradInput[imageOffset + i] = value;
                }
                gradInput[imageOffset + label] -= 1;
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            int label = labels[n];
            for(int plane = 0; plane < numPlanes; plane++) {
                float value = output[imageOffset + plane];
                if (std::isfinite(value) == false)
                    throw runtime_error("Output is a non-finite number, this usually means the learning rate is too high");
                gradInput[imageOffset + plane] = value;
            }
            if(label >= numPlanes) {
                throw runtime_error("Label " + toString(label) + " exceeds number of softmax planes " + toString(numPlanes) );
            } else if(label < 0) {
                throw runtime_error("Label " + toString(label) + " negative");
            }
            gradInput[imageOffset + label] -= 1;
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcGradInputfromlabels");
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayer::calcGradInput(float const *expectedValues) {
//    cout << "softmaxlayer::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer calcGradInput");
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                for(int i = 0; i < imageSizeSquared; i++) {
                    int resultIndex = imageOffset + i;
                    float value = output[resultIndex];
                    if (std::isfinite(value) == false)
                        throw runtime_error("Output is a non-finite number, this usually means the learning rate is too high");
                    gradInput[resultIndex] = value - expectedValues[resultIndex];
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            for(int plane = 0; plane < numPlanes; plane++) {
                int resultIndex = imageOffset + plane;
                float value = output[resultIndex];
                if (std::isfinite(value) == false)
                    throw runtime_error("Output is a non-finite number, this usually means the learning rate is too high");
                gradInput[resultIndex] = value - expectedValues[resultIndex];
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer calcGradInput");
}
VIRTUAL int SoftMaxLayer::getNumLabelsPerExample() {
    if(perPlane) {
        return numPlanes;
    } else {
        return imageSizeSquared;
    }
}
VIRTUAL int SoftMaxLayer::getPersistSize(int version) const {
    return 0;
}
VIRTUAL int SoftMaxLayer::calcNumRightFromLabels(int const*labels) {
    StatefulTimer::timeCheck("start SoftMaxLayer calcNumRight");
//    float *input = previousLayer->getOutput(); // just retrieve as host-side array for now
    int numRight = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                int label = labels[n * numPlanes + plane];
                float thisMax = output[imageOffset + 0];
                int iMax = 0;
                for(int i = 1; i < imageSizeSquared; i++) {
                    if(output[imageOffset + i] > thisMax) {
                        thisMax = output[imageOffset + i];
                        iMax = i;
                    }
                }
                if(label == iMax) {
//                    cout << "n " << n << " plane " << plane << " label " << label << endl;
                    numRight++;
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            int label = labels[n];
            float thisMax = output[imageOffset + 0];
            int iMax = 0;
            for(int i = 1; i < numPlanes; i++) {
                if(output[imageOffset + i] > thisMax) {
                    thisMax = output[imageOffset + i];
                    iMax = i;
                }
            }
            if(label == iMax) {
                numRight++;
            }
        }
    }

    StatefulTimer::timeCheck("start SoftMaxLayer calcNumRight");
    return numRight;
}
// for forward, we just need to apply the softmax activation. "just" :-P
VIRTUAL void SoftMaxLayer::forward() {
//    cout << "softmaxlayer::forward" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayer forward");
    float *input = previousLayer->getOutput(); // just retrieve as host-side array for now
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                float maxValue = input[imageOffset + 0];
                for(int i = 1; i < imageSizeSquared; i++) {
                    maxValue = std::max(maxValue, input[imageOffset + i]);
                }
                float denominator = 0;
                for(int i = 0; i < imageSizeSquared; i++) {
                    denominator += exp(input[imageOffset + i] - maxValue);
                }
                for(int i = 0; i < imageSizeSquared; i++) {
                    output[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            // first get the max
            float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
            for(int plane = 1; plane < numPlanes; plane++) {
                maxValue = std::max(maxValue, input[imageOffset + plane]);
            }
            // calculate sum, under this max
            float denominator = 0;
            for(int plane = 0; plane < numPlanes; plane++) {
                denominator += exp(input[imageOffset + plane] - maxValue);
            }
            // now calc the softmaxes:
            for(int plane = 0; plane < numPlanes; plane++) {
                output[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayer forward");
}
VIRTUAL void SoftMaxLayer::getLabels(int *labels) { // need to allocate labels array first, and have called 'forward' first
    if(perPlane) {
        throw std::runtime_error("getLabels doesnt work with 'perPlane' option currently, though it wouldnt be hard to add, so ask if you need");
    }
    if(imageSize != 1) {
        throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
    }
    for(int n = 0; n < batchSize; n++) {
        float *outputStack = output + n * numPlanes;
        float highestProb = outputStack[0];
        int bestPlane = 0;
        for(int plane = 1; plane < numPlanes; plane++) {
            if(outputStack[plane] > highestProb) {
                bestPlane = plane;
                highestProb = outputStack[plane];
            }
        }
        labels[n] = bestPlane;
    }
}
// this seems to be handled by calcGradInput? So, just to a nop?
// (cos this layer kind of combines loss layer and a 'normal' propagation layer)
// certainly, we dont have any weights to update, and we already handled error
// propagation in 'calcGradInput' method above
//VIRTUAL void SoftMaxLayer::backward(float learningRate) {
//    cout << "softmaxlayer::backproperrors" << endl;
    // nop, do nothing :-)
//}
VIRTUAL std::string SoftMaxLayer::asString() const {
    return "SoftMaxLayer{ perPlane=" + toString(perPlane) + " numPlanes=" + toString(numPlanes)
        + " imageSize=" + toString(imageSize) + " }";
}

