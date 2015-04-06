#!/usr/bin/python

from __future__ import print_function

#from array import array
import sys
import array
import PyDeepCL

class ScenarioImage(PyDeepCL.Scenario):
    def __init__(self, size, apple_moves):
        super(ScenarioImage,self).__init__()
        self.size = size
        self.apple_moves = apple_moves
    def getPerceptionSize(self):
        return self.size
    def getNumActions(self):
        return 4
    def getPerceptionPlanes(self):
        return 2
    def getPerception(self):
        return [0] * self.getPerceptionPlanes() * self.getPerceptionSize()
    def act(self,index):
        print('pretending to act :-)')
        return 0
    def hasFinished(self):
        return False
    def show(self):
        print('showing')

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

