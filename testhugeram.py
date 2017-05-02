import PyDeepCL
import array

targets = array.array('f', output_data)
images = array.array('f', input_data)

net.setBatchSize(batchSize)
im1 = []
for epoch in range(0, numEpochs):
    print 'epoch', epoch
    context = PyDeepCL.TrainingContext(epoch, 0)
    for batch in range(N / batchSize):
        sgd.train(
            net,
            context,
            images[batch * batchSize * planes * size * size:],
            targets[batch * batchSize * planes:])

        net.forward(images[0:(batchSize * planes * size * size)])
        if batch == 0:
            lastLayer = net.getLastLayer()
            predictions = lastLayer.getOutput()[0:(planes * batchSize)]
            precision = np.mean(np.sqrt((np.array(predictions) - output_data[0:(planes * batchSize)]) ** 2))
            print precision
            if epoch == numEpochs - 1:
                for i in range(8):
                    im1.extend(predictions[((i * nt) * planes):(((i * nt) + 1) * planes) ])

