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

function test_genericloader()
    require('luaDeepCL')

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

os.exit( luaunit.LuaUnit.run() )

