#include <iostream>

#include "deepcl/DeepCL.h"

int main(int argc, char *argv[])
{
    EasyCL *cl = new EasyCL();
    NeuralNet *net = new NeuralNet(cl);
    Trainer *trainer = SGD::instance( cl, 0.05f, 0.0f );

    net->addLayer( InputLayerMaker::instance()->numPlanes(2)->imageSize(1) );
    net->addLayer( FullyConnectedMaker::instance()->numPlanes(2)->imageSize(1) );
    net->addLayer( ActivationMaker::instance()->sigmoid() );
    net->addLayer( FullyConnectedMaker::instance()->numPlanes(1)->imageSize(1) );
    net->addLayer( ActivationMaker::instance()->tanh() );
    net->addLayer( SquareLossMaker::instance() );


    // ExpectedData train
    for (int var = 0; var < 500; ++var) {
        //random error occurred when batchSize over 64
        //I don't know how to determine the maximum of batchSize
        int batchSize = 32;
        float input[batchSize*2] = {0};
        float expectedOutput[batchSize] = {0};

        for (int batch = 0; batch < batchSize; ++batch) {
            int a =rand()%2;
            int b =rand()%2;

            input[batch*2] = a? 1: -1;
            input[batch*2+1] = b? 1: -1;

            expectedOutput[batch] = a ^ b;
        }

        TrainingContext tc(var, var);
        net->setBatchSize(batchSize);
        BatchResult batchResult = trainer->trainNet(net, &tc, input, expectedOutput);

        if(var%100 == 0)
            std::cout << var << "... Loss:" << batchResult.getLoss() << std::endl;
    }


    // Test the net
    std::cout << "ExpectedData Test:" << std::endl;
    int numExamples = 4;
    float data[] = { -1, -1,
                     -1, 1,
                     1, -1,
                     1, 1 };

    net->setBatchSize(numExamples);
    net->forward(data);

    const float * output = net->getOutput();
    for (int i = 0; i < numExamples; ++i) {
        std::cout << data[i*2] << " xor " << data[i*2+1] << " = " << output[i] << std::endl;
    }

    delete trainer;
    delete net;
    delete cl;

    return 0;
}

