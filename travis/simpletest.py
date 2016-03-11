"""
simple smoke test that can at least load DeepCL, run a couple of simple fucntions.  To be used in the 
travis builds at https://travis-ci.org/hughperkins/DeepCL/builds
"""

from __future__ import print_function, division
import array
import PyDeepCL
import sys
import numpy as np

cl = PyDeepCL.DeepCL()

print('compute units:', cl.getComputeUnits())
print('local memory size, bytes:', cl.getLocalMemorySize())
print('local memory size, KB:', cl.getLocalMemorySizeKB())
print('max workgroup size:', cl.getMaxWorkgroupSize())
print('max alloc size MB:', cl.getMaxAllocSizeMB())

batchSize = 2
numPlanes = 1
imageSize = 6

net = PyDeepCL.NeuralNet(cl)
net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(1).imageSize(6))
net.addLayer(PyDeepCL.NormalizationLayerMaker().translate(-0.5).scale(1/255.0))
net.addLayer(PyDeepCL.ActivationMaker().relu())
net.addLayer(PyDeepCL.PoolingMaker().poolingSize(2))

inTensor = np.random.randn(batchSize * numPlanes * imageSize * imageSize).astype('float32')
net.setBatchSize(batchSize)
net.forward(inTensor)
outTensor = net.getOutput()
print('len(outTensor)', len(outTensor))
assert(len(outTensor) == batchSize * numPlanes * (imageSize//2) * (imageSize//2))
print('finished ok')

