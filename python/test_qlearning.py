#!/usr/bin/python

from __future__ import print_function

#from array import array
import sys
import array
import random
import PyDeepCL

# This is an example scenario.  It overrides the PyDeepCL.Scenario class
# The Q-learning module will call into this object each time it makes a move
# This class can therefore represent any world you want to expose to the
# q-learning module
class ScenarioImage(PyDeepCL.Scenario):
    def __init__(self, size, apple_moves):
        """Standard constructor.  Do whatever you need to set up the world"""
        super(ScenarioImage,self).__init__()
        self.size = size
        self.appleMoves = apple_moves
        self.finished = False
        self.game = 0
        self.reset()
    def getPerceptionSize(self):
        """Assumes perception is square.  This is the length of one edge"""
        return self.size
    def getNumActions(self):
        """How many possible virtual 'buttons' can the computer push?"""
        return 4
    def getPerceptionPlanes(self):
        """We can feed one or more planes to the qleaning module"""
        return 2
    def getPerception(self):
        """Need to provide the current perception to the qlearning module,
        which should be of size numPlanes * size * size"""
        perception = [0] * 2 * self.size * self.size
        perception[self.appleY * self.size + self.appleX] = 1;
        perception[self.size * self.size + self.posY * self.size + self.posX] = 1; 
        return perception
    def act(self,index):
        """The computer chooses one of the numActions available actions
        this method needs to update the world, and return the reward
        (positive or negative)"""
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
        """If the last action ended this particular game/world-instance
        then this should return True.  After 'reset' has been called
        it should return False again"""
        #print('scenarioimage.hasFinished()') 
        return self.finished
    def setNet(self, net):
        """This doesnt override anything from the base class, we're simply using 
        it, because then we can use it to print a q representation, eg at the
        end of each game"""
        self.net = net
    def _show(self):
        """can do nothing, or it can print the world somehow.
        This provides no information to the qlearning module: it's
        simply an opportunity for you to see how the world looks
        occasionally"""
        print('pos',self.posX,self.posY,'apple',self.appleX,self.appleY)
        for y in range(self.size):
            line = ''
            for x in range(self.size):
                if x == self.posX and y == self.posY:
                    line += "X"
                elif x == self.appleX and y == self.appleY:
                    line += "O"
                else:
                    line += "."
            print(line)
    def _showQ(self):
        """can do nothing, or it can print the current q 
        values somehow.
        This provides no information to the qlearning module: it's
        simply an opportunity for you to see how the q value look
        occasionally"""
#        print('showQ()')
#        print('net num layers: ' + str(net.getNumLayers() ) ) # proves we do have a copy of the network :-)
        scenario = self
        net = self.net
        print( "q directions:" )
        size = self.size
        netinput = array.array( 'f', [0] * (2*size*size) )
        netinput[ self.appleY * size + self .appleX ] = 1
        for y in range(size):
            thisLine = ''
            for x in range(size):
                highestQ = 0
                bestAction = 0
                netinput[ size * size + y * size + x ] = 1
                net.propagate( netinput )
                netinput[ size * size + y * size + x ] = 0
                results = net.getResults()
                for action in range(4):
                    thisQ = results[action]
                    if action == 0 or thisQ > highestQ:
                        highestQ = thisQ
                        bestAction = action
                if bestAction == 0:
                    thisLine += ">"
                elif bestAction == 1:
                    thisLine += "<"
                elif bestAction == 2:
                    thisLine += "V"
                else:
                    thisLine += "^"
            print(thisLine)
    def reset(self):
        """starts a new game / world-instance"""
        # first, lets print the final world and q-state:
        # this used to be called by the qlearning module
        # but seems to make more sense - and be more 
        # flexible :-) - to call it from here, ourselves
        # we can then call it ourselves from 'act' etc
        # too, if we wish
        if self.game >= 1:
            self._show()
            self._showQ()
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
        self.game += 1

def go():
    """creates a net, instantiates the scenario, and calls into the qlearning
    module, to start learning"""

    scenario = ScenarioImage(5,True)

    size = scenario.getPerceptionSize();
    planes = scenario.getPerceptionPlanes();
    numActions = scenario.getNumActions();
    #size = 5
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

    scenario.setNet(net)

    qlearner = PyDeepCL.QLearner( scenario, net )
    # qlearner.setLambda(0.9) # sets decay of the eligibility trace decay rate
    # qlearner.setMaxSamples(32) # how many samples to learn from after each move
    # qlearner.setEpsilon(0.1) # probability of exploring, instead of exploiting
    # qlearner.setLearningRate(0.1) # learning rate of the neural net
    qlearner.run()

if __name__ == '__main__':
    go()


