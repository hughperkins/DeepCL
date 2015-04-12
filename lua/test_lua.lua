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

luaunit = require('thirdparty.luaunit')
require('luaDeepCL')
deepcl = luaDeepCL

function test_genericloader()

    deepcl = luaDeepCL

    genericLoader = deepcl.GenericLoader()

    trainfilepath = '../data/mnist/train-images-idx3-ubyte'
    N,planes,size = deepcl.GenericLoader_getDimensions( trainfilepath )
    print('N='..N..' planes='..planes..' size='..size)

    N = 10
    images = deepcl.floatArray(N * planes * size * size )
    labels = deepcl.intArray(N)
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

    deepcl = luaDeepCL

    genericLoader = deepcl.GenericLoader()

    trainfilepath = '../data/mnist/train-images-idx3-ubyte'
    N,planes,size = deepcl.GenericLoader_getDimensions( trainfilepath )
    print('N='..N..' planes='..planes..' size='..size)

    N = 1280

    images = deepcl.floatArray(N * planes * size * size )
    labels = deepcl.intArray(N)
    deepcl.GenericLoader_load( trainfilepath, images, labels, 0, N )
    print('images',images)
    print('labels',labels)

    net = deepcl.NeuralNet(1,28)
    print(net:asString())
    normmaker = deepcl.NormalizationLayerMaker()
    net:addLayer( deepcl.NormalizationLayerMaker():translate(-40):scale(1/255.0) )
    print(net:asString())
    deepcl.NetdefToNet_createNetFromNetdef( net, "rt2-8c5-mp2-16c5-mp3-150n-10n" ) 
    print(net:asString())

    learner = deepcl.NetLearnerFloats( net )
    learner:setTrainingData( N, images, labels )
    learner:setTestingData( N, images, labels )
    learner:setSchedule( 12 )
    learner:setBatchSize( 128 )
    learner:learn( 0.002 )
end

os.exit( luaunit.LuaUnit.run() )

