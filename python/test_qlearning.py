#!/usr/bin/python

from __future__ import print_function

#from array import array
import sys
import array
import random
import PyDeepCL

class ScenarioImage(PyDeepCL.Scenario):
    def __init__(self, size, apple_moves):
        super(ScenarioImage,self).__init__()
        self.size = size
        self.appleMoves = apple_moves
        self.finished = False
        self.reset()
    def getPerceptionSize(self):
        return self.size
    def getNumActions(self):
        return 4
    def getPerceptionPlanes(self):
        return 2
    def getPerception(self):
        perception = [0] * 2 * self.size * self.size
        perception[self.appleY * self.size + self.appleX] = 1;
        perception[self.size * self.size + self.posY * self.size + self.posX] = 1; 
        return perception
    def act(self,index):
#        print('pretending to act :-)')
        dx = 0
        dy = 0
        if index == 0:
            dx = 1
        elif index == 1:
            dx = -1
        elif index == 2:
            dy = 1
        elif index == 3:
            dy = -1
        newX = self.posX + dx;
        newY = self.posY + dy;
        if newX < 0 or newX >= self.size or newY < 0 or newY >= self.size:
            return -0.5
        if newX == self.appleX and newY == self.appleY:
            self.finished = True
            self.posX = newX
            self.posY = newY
            return 1
        else:
            self.posX = newX
            self.posY = newY
            return -0.1
    def hasFinished(self):
        #print('scenarioimage.hasFinished()') 
        return self.finished
    def show(self):
#        print('showing')
        print('pos',self.posX,self.posY,'apple',self.appleX,self.appleY)
    def showQ(self,net):
        print('showQ()')
        print('net num layers: ' + str(net.getNumLayers() ) ) # proves we do have a copy of the network :-)
    def reset(self):
        print('scenarioimage.reset()') 
        if self.appleMoves:
            self.appleX = random.randint(0, self.size-1)
            self.appleY = random.randint(0, self.size-1)
        else:
            self.appleX = self.appleY = self.size // 2
        self.finished = False
        sampledOnce = False
        while not sampledOnce or ( self.posX == self.appleX and self.posY == self.appleY ):
            self.posX = random.randint(0, self.size-1)
            self.posY =random.randint(0, self.size-1)
            sampledOnce = True

def go():
    scenario = ScenarioImage(5,False)

    size = scenario.getPerceptionSize();
    planes = scenario.getPerceptionPlanes();
    numActions = scenario.getNumActions();
    #size = 5
    #numActions = 4
    #planes = 2
    print('size',size,'planes',planes,'numActions',numActions)

    net = PyDeepCL.NeuralNet()
    net.addLayer( PyDeepCL.InputLayerMaker().numPlanes(planes).imageSize(size) )
    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().relu() )
    net.addLayer( PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased().relu() )
    net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(100).imageSize(1).biased().tanh() )
    net.addLayer( PyDeepCL.FullyConnectedMaker().numPlanes(numActions).imageSize(1).biased().linear() )
    net.addLayer( PyDeepCL.SquareLossMaker() )
    print( net.asString() )

    qlearner = PyDeepCL.QLearner( scenario, net )
    qlearner.run()

if __name__ == '__main__':
    go()

