from __future__ import print_function, division

import platform
import array
import PyDeepCL
import numpy as np

if __name__ == '__main__':
    N = 1280
    batchSize = 128
    planes = 1
    size = 28
    numEpochs = 2000

    output_data = np.random.rand(N * planes * size * size).astype('float32')
    input_data = np.random.rand(N * planes * size * size).astype('float32')
    targets = array.array('f', input_data)
    images = array.array('f', output_data)

    test_input = images[0:(batchSize * planes * size * size)]

    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(planes).imageSize(size))
#    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(20).filterSize(3).
#                 biased())
#    net.addLayer(PyDeepCL.ActivationMaker().relu())
#    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(49).filterSize(1).
#                 biased())
    net.addLayer(PyDeepCL.SquareLossMaker())
    sgd = PyDeepCL.SGD(cl, 0.00002, 0.0001)
    print(net.asString())

    net.setBatchSize(batchSize)
    im1 = []

#    imageBatches = []
#    targetBatches = []
#    for batch in range(N // batchSize):
#        imageBatches.append(images[batch * batchSize * planes * size * size:])
#        targetBatches.append(targets[batch * batchSize * planes:])

    for epoch in range(0, numEpochs):
        print('epoch', epoch)
        context = PyDeepCL.TrainingContext(epoch, 0)
        for batch in range(N // batchSize):
#            sgd.train(
#                net,
#                context,
#                input_data[batch],
#                output_data[batch])

            test_input = input_data[0:(batchSize * planes * size * size)]
            net.forward(test_input)
#            if batch == 0:
#                lastLayer = net.getLastLayer()
#                predictions = lastLayer.getOutput()[0:(planes * batchSize)]
#                precision = np.mean(np.sqrt((np.array(predictions) \
#                                - output_data[0:(planes * batchSize)]) ** 2))
#                print precision
        if epoch % 20 == 0:
            pyversion = int(platform.python_version_tuple()[0])
            print('pyversion', pyversion)
            if pyversion == 2:
                print('py 2')
                raw_input(str(epoch))
            else:
                print('py 3')
                input(str(epoch))

