#include <iostream>

#include "deepcl/DeepCL.h"

int main(int argc, char *argv[])
{
    EasyCL *cl = new EasyCL();
    Trainer *trainer = SGD::instance( cl, 0.1f, 0.0f );

    NeuralNet *net = new NeuralNet(cl);
    net->addLayer( InputLayerMaker::instance()->numPlanes(2)->imageSize(1) );
    net->addLayer( ConvolutionalMaker::instance()->numFilters(4)->filterSize(1)->padZeros()->biased() );
    net->addLayer( ActivationMaker::instance()->relu() );
    net->addLayer( FullyConnectedMaker::instance()->numPlanes(2)->imageSize(1) );
    net->addLayer( ActivationMaker::instance()->sigmoid() );
    net->addLayer( SoftMaxMaker::instance() );
    net->print();

    int numExamples = 4;

    float data[] = { -1, -1,
                     -1, 1,
                     1, -1,
                     1, 1 };

    int labels[] = {
        0,
        1,
        1,
        0
    };

    NetLearner netLearner(trainer, net, numExamples, data, labels, numExamples, data, labels, numExamples);
    netLearner.reset();
    netLearner.setSchedule(500);
    netLearner.run();

    return 0;
}
