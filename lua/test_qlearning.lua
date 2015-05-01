-- Copyright Hugh Perkins 2015 hughperkins at gmail

-- This Source Code Form is subject to the terms of the Mozilla Public License, 
-- v. 2.0. If a copy of the MPL was not distributed with this file, You can 
-- obtain one at http://mozilla.org/MPL/2.0/.

-- test the wrappers
-- to run:
-- 
--     LD_LIBRARY_PATH=../build:. luajit test_qlearning.lua
--

-- luaunit = require('thirdparty.luaunit')
require('LuaDeepCL')
deepcl = LuaDeepCL

function zeroArray( N, floatarray )
    for i = 0, N-1 do
        floatarray[i] = 0
    end
end

ScenarioImage = {} 
function ScenarioImage.new(parent, size, appleMoves)
    local self = {size=size, appleMoves=appleMoves}
    setmetatable( self, {__index=parent})
    self.size = size
    self.appleMoves = appleMoves
    self.finished = false
    self.numActions = 4
    self.game = 0
    --self:reset()
    return self
end
function ScenarioImage.getPerceptionSize(self)
    -- Assumes perception is square.  This is the length of one edge
    return self.size
end
function ScenarioImage.getNumActions(self)
    -- How many possible virtual 'buttons' can the computer push?
    return 4
end
function ScenarioImage.getPerceptionPlanes(self)
    -- We can feed one or more planes to the qleaning module
    return 2
end
function ScenarioImage.getPerception(self, perception)
    -- Need to provide the current perception to the qlearning module,
    -- which should be of size numPlanes * size * size
    --perception = [0] * 2 * self.size * self.size
    zeroArray( 2 * self.size * self.size, perception )
    perception[self.appleY * self.size + self.appleX] = 1
    perception[self.size * self.size + self.posY * self.size + self.posX] = 1
end
function ScenarioImage.act(self, action)
    -- The computer chooses one of the numActions available actions
    -- this method needs to update the world, and return the reward
    -- (positive or negative)
    self.numMoves = self.numMoves + 1
    dx = 0
    dy = 0
    if action == 0 then
        dx = 1
    elseif action == 1 then
        dx = -1
    elseif action == 2 then
        dy = 1
    elseif action == 3 then
        dy = -1
    end
    newX = self.posX + dx
    newY = self.posY + dy
    if newX < 0 or newX >= self.size or newY < 0 or newY >= self.size then
        return -0.5
    end
    if newX == self.appleX and newY == self.appleY then
        self.finished = true
        self.posX = newX
        self.posY = newY
        return 1
    else
        self.posX = newX
        self.posY = newY
        return -0.1
    end
end
function ScenarioImage.hasFinished(self)
    -- If the last action ended this particular game/world-instance
    -- then this should return True.  After 'reset' has been called
    -- it should return False again
    -- print('scenarioimage.hasFinished()') 
    return self.finished
end
function ScenarioImage.show(self)
    -- print the world
    print('pos '..self.posX ..', '..self.posY .. ' apple ' ..self.appleX..', '..self.appleY)
    for y = 0, self.size - 1 do
        local line = ''
        for x = 0, self.size - 1 do
            if x == self.posX and y == self.posY then
                line = line .. "X"
            elseif x == self.appleX and y == self.appleY then
                line = line .. "O"
            else
                line = line .. "."
            end
        end
        print(line)
    end
end
function ScenarioImage.showQ(self, net)
    -- print the current q 
    -- values somehow.
    local scenario = self
--    local net = self.net
    print( "q directions:" )
    local size = self.size
    local netinput = deepcl.floatArray(2*size*size)
    zeroArray( 2 * size * size, netinput )
    netinput[ self.appleY * size + self .appleX ] = 1
    for y = 0, size-1 do
        local thisLine = ''
        for x = 0, size - 1 do
            local highestQ = 0
            local bestAction = 0
            netinput[ size * size + y * size + x ] = 1
            net:setBatchSize(1)
            net:forward( netinput )
            netinput[ size * size + y * size + x ] = 0
            local output = deepcl.floatArray(self.numActions)
            net:getOutput(output)
            for action = 0, 3 do
                local thisQ = output[action]
                if action == 0 or thisQ > highestQ then
                    highestQ = thisQ
                    bestAction = action
                end
            end
            if bestAction == 0 then
                thisLine = thisLine .. ">"
            elseif bestAction == 1 then
                thisLine = thisLine .. "<"
            elseif bestAction == 2 then
                thisLine = thisLine .. "V"
            else
                thisLine = thisLine .. "^"
            end
        end
        print(thisLine)
    end
end
function ScenarioImage.getNumMoves(self)
    return self.numMoves
end
function ScenarioImage.reset(self)
--    print('ScenarioImage.reset()')
    -- starts a new game / world-instance"
    if self.appleMoves then
        self.appleX = math.random(0, self.size-1)
        self.appleY = math.random(0, self.size-1)
    else
        self.appleX = math.floor(self.size / 2)
        self.appleY = math.floor(self.size / 2)
    end
--    print('apple ' .. self.appleX..', '..self.appleY)
    self.finished = false
    local sampledOnce = false
    while not sampledOnce or ( self.posX == self.appleX and self.posY == self.appleY ) do
        self.posX = math.random(0, self.size-1)
        self.posY = math.random(0, self.size-1)
        sampledOnce = true
    end
    self.game = self.game + 1
    self.numMoves = 0
end
function go()
    -- creates a net, instantiates the scenario, and calls into the qlearning
    -- module, to start learning

    local size = 5
    local appleMoves = true
    local scenario = ScenarioImage:new(size, appleMoves)
    local planes = scenario:getPerceptionPlanes()
    local numActions = scenario:getNumActions()
    print('size',size,'planes',planes,'numActions',numActions)

    local cl = deepcl.EasyCL()
    local net = deepcl.NeuralNet(cl)
    local sgd = deepcl.SGD.instance(cl, 0.1, 0.0)
    net:addLayer( deepcl.InputLayerMaker():numPlanes(planes):imageSize(size) )
    net:addLayer( deepcl.ConvolutionalMaker():numFilters(8):filterSize(5):padZeros():biased() )
    net:addLayer( deepcl.ActivationMaker():relu() )
    net:addLayer( deepcl.ConvolutionalMaker():numFilters(8):filterSize(5):padZeros():biased() )
    net:addLayer( deepcl.ActivationMaker():relu() )
    net:addLayer( deepcl.FullyConnectedMaker():numPlanes(100):imageSize(1):biased() )
    net:addLayer( deepcl.ActivationMaker():tanh() )
    net:addLayer( deepcl.FullyConnectedMaker():numPlanes(numActions):imageSize(1):biased() )
    net:addLayer( deepcl.SquareLossMaker() )
    print( net:asString() )

    qlearner = deepcl.QLearner2(sgd, net, numActions, planes, size)
    -- qlearner:setLambda(0.9) # sets decay of the eligibility trace decay rate
    -- qlearner:setMaxSamples(32) # how many samples to learn from after each move
    -- qlearner:setEpsilon(0.1) # probability of exploring, instead of exploiting
    -- qlearner:setLearningRate(0.1) # learning rate of the neural net

    game = 0;
    local lastReward = 0
    local perception = deepcl.floatArray( size * size * planes )
    local wasReset = false
    scenario:reset()
    while true do
        scenario:getPerception( perception )
        local action = qlearner:step( lastReward, wasReset, perception )
        lastReward = scenario:act( action )
        if scenario:hasFinished() then
            scenario:show()
            scenario:showQ(net)
            print('game='..game.. ' numMoves='.. scenario:getNumMoves())
            scenario:reset()
            wasReset = true
            game = game + 1
        else
            wasReset = false
        end
    end
end


go()

-- os.exit( luaunit.LuaUnit.run() )

