#!/usr/bin/python

from __future__ import print_function
import numpy as np
import random
import time
import PyDeepCL


class ScenarioImage(PyDeepCL.Scenario):
    """
    This is an example scenario.  It overrides the PyDeepCL.Scenario class
    The Q-learning module will call into this object each time it makes a move
    This class can therefore represent any world you want to expose to the
    q-learning module
    """
    def __init__(self, size, apple_moves):
        """
        Standard constructor.  Do whatever you need to set up the world
        """
        super(ScenarioImage, self).__init__()
        self.size = size
        self.appleMoves = apple_moves
        self.finished = False
        self.game = 0
        self.last = time.time()
        self.reset()
        self.last = time.time()
        self.perception = np.zeros((2, size, size), dtype=np.float32)
        self.netinput = np.zeros((2, size, size), dtype=np.float32)

    def getPerceptionSize(self):
        """
        Assumes perception is square.  This is the length of one edge
        """
        return self.size

    def getNumActions(self):
        """
        How many possible virtual 'buttons' can the computer push?
        """
        return 4

    def getPerceptionPlanes(self):
        """
        We can feed one or more planes to the qleaning module
        """
        return 2

    def getPerception(self):
        """
        Need to provide the current perception to the qlearning module,
        which should be of size numPlanes * size * size
        """
        self.perception.fill(0)
        self.perception[0, self.appleY, self.appleX] = 1
        self.perception[1, self.posY, self.posX] = 1
        # print(self.appleY, self.appleX, self.posY, self.posX)
        if time.time() - self.last > 1.0:
            print('round: %s' % self.game)
            self._show()
            self._showQ()
            self.last = time.time()
        return self.perception

    def act(self, index):
        """
        The computer chooses one of the numActions available actions
        this method needs to update the world, and return the reward
        (positive or negative)
        """
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
        newX = self.posX + dx
        newY = self.posY + dy
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
        """
        If the last action ended this particular game/world-instance
        then this should return True.  After 'reset' has been called
        it should return False again
        """
        return self.finished

    def setNet(self, net):
        """
        This doesnt override anything from the base class, we're simply using
        it, because then we can use it to print a q representation, eg at the
        end of each game
        """
        self.net = net

    def _show(self):
        """
        can do nothing, or it can print the world somehow.
        This provides no information to the qlearning module: it's
        simply an opportunity for you to see how the world looks
        occasionally
        """
        print('pos', self.posX, self.posY, 'apple', self.appleX, self.appleY)
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
        """
        can do nothing, or it can print the current q
        values somehow.
        This provides no information to the qlearning module: it's
        simply an opportunity for you to see how the q value look
        occasionally
        """
        net = self.net
        print("q directions:")
        size = self.size
        self.netinput.fill(0)
        self.netinput[0, self.appleY, self.appleX] = 1
        for y in range(size):
            thisLine = ''
            for x in range(size):
                highestQ = 0
                bestAction = 0
                self.netinput[1, y, x] = 1
                # netinput[size * size + y * size + x] = 1
                net.forward(self.netinput)
                self.netinput[1, y, x] = 0
                # netinput[size * size + y * size + x] = 0
                output = net.getOutput()
                for action in range(4):
                    thisQ = output[action]
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
        """
        starts a new game / world-instance
        first, lets print the final world and q-state:
        this used to be called by the qlearning module
        but seems to make more sense - and be more
        flexible :-) - to call it from here, ourselves
        we can then call it ourselves from 'act' etc
        too, if we wish
        """
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
        while not sampledOnce or (
                self.posX == self.appleX and self.posY == self.appleY):
            self.posX = random.randint(0, self.size-1)
            self.posY = random.randint(0, self.size-1)
            sampledOnce = True
        self.game += 1


def go():
    """
    creates a net, instantiates the scenario, and calls into the qlearning
    module, to start learning
    """

    scenario = ScenarioImage(5, True)

    size = scenario.getPerceptionSize()
    planes = scenario.getPerceptionPlanes()
    numActions = scenario.getNumActions()
    print('size', size, 'planes', planes, 'numActions', numActions)

    cl = PyDeepCL.DeepCL()
    net = PyDeepCL.NeuralNet(cl)
    sgd = PyDeepCL.SGD(cl, 0.02, 0.0)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(planes).imageSize(size))
    net.addLayer(
        PyDeepCL.ConvolutionalMaker()
        .numFilters(8).filterSize(3).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().relu())
    net.addLayer(
        PyDeepCL.ConvolutionalMaker()
        .numFilters(8).filterSize(3).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().relu())
    net.addLayer(
        PyDeepCL.FullyConnectedMaker().numPlanes(100).imageSize(1).biased())
    net.addLayer(PyDeepCL.ActivationMaker().tanh())
    net.addLayer(
        PyDeepCL.FullyConnectedMaker()
        .numPlanes(numActions).imageSize(1).biased())
    net.addLayer(PyDeepCL.SquareLossMaker())
    print(net.asString())

    scenario.setNet(net)

    qlearner = PyDeepCL.QLearner(sgd, scenario, net)
    # sets decay of the eligibility trace decay rate
    qlearner.setLambda(0.9)
    # how many samples to learn from after each move
    qlearner.setMaxSamples(32)
    # probability of exploring, instead of exploiting
    qlearner.setEpsilon(0.1)
    qlearner.run()

if __name__ == '__main__':
    go()
