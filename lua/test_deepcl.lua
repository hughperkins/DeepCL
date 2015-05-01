-- Copyright Hugh Perkins 2015 hughperkins at gmail

-- This Source Code Form is subject to the terms of the Mozilla Public License, 
-- v. 2.0. If a copy of the MPL was not distributed with this file, You can 
-- obtain one at http://mozilla.org/MPL/2.0/.

-- test the wrappers
-- to run:
-- 
--     LD_LIBRARY_PATH=../build:. luajit test_lua.lua
--

print('test_lua.lua')

luaunit = require('luaunit')
require('LuaDeepCL')
deepcl = LuaDeepCL

function test_genericloader()

    local deepcl = LuaDeepCL

    -- local genericLoader = deepcl.GenericLoader()

    local trainfilepath = '../data/mnist/train-images-idx3-ubyte'
    local N,planes,size = deepcl.GenericLoader_getDimensions( trainfilepath )
    print('N='..N..' planes='..planes..' size='..size)

    local N = 10
    local images = deepcl.floatArray(N * planes * size * size )
    local labels = deepcl.intArray(N)
    deepcl.GenericLoader_load( trainfilepath, images, labels, 0, 10 )
    print('images',images)
    print('labels',labels)
    for i = 0,9 do
        print(i,labels[i])
    end
    luaunit.assertEquals(labels[0], 5)
    luaunit.assertEquals(labels[1], 0)
    luaunit.assertEquals(labels[5], 2)
    luaunit.assertEquals(labels[9], 4)
end

function test_basic()

    local deepcl = LuaDeepCL

    -- local genericLoader = deepcl.GenericLoader()

    local trainfilepath = '../data/mnist/train-images-idx3-ubyte'
    local N,planes,size = deepcl.GenericLoader_getDimensions( trainfilepath )
    print('N='..N..' planes='..planes..' size='..size)

    local N = 1280

    local images = deepcl.floatArray(N * planes * size * size )
    local labels = deepcl.intArray(N)
    deepcl.GenericLoader_load( trainfilepath, images, labels, 0, N )
    print('images',images)
    print('labels',labels)

    local cl = deepcl.EasyCL()
    local net = deepcl.NeuralNet(cl, 1,28)
    local sgd = deepcl.SGD_instance(cl, 0.002, 0)
    print(net:asString())
    net:addLayer( deepcl.NormalizationLayerMaker():translate(-40):scale(1/255.0) )
    print(net:asString())
    deepcl.NetdefToNet_createNetFromNetdef( net, "rt2-8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n" ) 
    print(net:asString())

    local learner = deepcl.NetLearner(
        sgd, net,
        N, images, labels,
        N, images, labels,
        128 )
    learner:setSchedule( 12 )
    learner:run()
end

function test_lowlevel()
    local N = 1280
    local batchSize = 128
    local numEpochs = 30

    local cl = deepcl.EasyCL()
    local net = deepcl.NeuralNet(cl)

    net:addLayer( deepcl.InputLayerMaker():numPlanes(1):imageSize(28) )
    net:addLayer( deepcl.NormalizationLayerMaker():translate(-0.5):scale(1/255.0) )
    net:addLayer( deepcl.ConvolutionalMaker():numFilters(8):filterSize(5):padZeros():biased() )
    net:addLayer( deepcl.ActivationMaker():relu() )
    net:addLayer( deepcl.PoolingMaker():poolingSize(2) )
    net:addLayer( deepcl.ConvolutionalMaker():numFilters(8):filterSize(5):padZeros():biased() )
    net:addLayer( deepcl.ActivationMaker():relu() )
    net:addLayer( deepcl.PoolingMaker():poolingSize(3) )
    net:addLayer( deepcl.FullyConnectedMaker():numPlanes(150):imageSize(1):biased() )
    net:addLayer( deepcl.ActivationMaker():tanh() )
    net:addLayer( deepcl.FullyConnectedMaker():numPlanes(10):imageSize(1):biased() )
    print( net:asString() )
    -- net.addLayer( deepcl.SquareLossMaker() )
    net:addLayer( deepcl.SoftMaxMaker() )
    print('created net')
    print( net:asString() )

    local trainfilepath = '../data/mnist/train-images-idx3-ubyte'
    local N,planes,size = deepcl.GenericLoader_getDimensions( trainfilepath )
    print('N='..N..' planes='..planes..' size='..size)
    local N = 1280
    local images = deepcl.floatArray(N * planes * size * size )
    local labels = deepcl.intArray(N)
    deepcl.GenericLoader_load( trainfilepath, images, labels, 0, N )
    print('images',images)
    print('labels',labels)
    print('got data')

    local sgd = deepcl.SGD_instance(cl, 0.002, 0)
    sgd:setWeightDecay(0.0001)
    net:setBatchSize(batchSize)
    for epoch = 0,numEpochs-1 do
        print('epoch', epoch)
        local numRight = 0
        for batch = 0, math.floor( N /  batchSize ) - 1 do
            imagesslice = deepcl.floatSlice( images, batch * batchSize * planes * size * size )
            labelsslice = deepcl.intSlice( labels, batch * batchSize )
            sgd:trainFromLabels(net, imagesslice, labelsslice)
--            net:forward( imagesslice )
--            net:backwardFromLabels( labelsslice )
            numRight = numRight + net:calcNumRight( labelsslice )
            -- print( 'numright ' + str( net:calcNumRight( labels ) ) )
        end
        print( 'num right: '..numRight )
    end
end

os.exit( luaunit.LuaUnit.run() )

