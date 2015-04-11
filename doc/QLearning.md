# Q-learning

Q-learning can be used:
* from C++
* from Python

## Concepts

Q-learning places the agent in an entirely unknown environment, and the agent needs to learn to maximize its reward.  Well, it's not quite entirely unknown environment.  The agent knows:
* it has some buttons to push, though it doesnt know what they can do
* it has a view of the current environment, available as an image, of one or more planes
* and that it wants to receive as much reward as possible, after each action

More concepts are described very well in [Lin's 1993 thesis](http://www.dtic.mil/dtic/tr/fulltext/u2/a261434.pdf).

The q-learning implementation implements experience replay (parameterized by `maxSamples`), and will act in an environment where the agent can 'see' an image, which updates after each `act`.  The image is the `perception`, and can have one or more planes.  Each move the agent will `act`, and be rewarded appropriately.

We write a Scenario implementation, which inherits from the Scenario class, and override the `act` and `getPerception` methods to return these to the agent.  `act` should return the reward, as a float. `getPerception` should return an array of floats, corresponding to the planes of images, ordered as: plane, row, column.  ie, point [plane][row][col] should be at [plane * numrows * numcols + row * numcols + col].

There are a couple of additional methods so the agent can determine how many actions there are (numbered from 0, sequentially), how many planes in the perception, and how big is the perception.   Perception is square for now. 

## C++ demo

You can see a C++ demo at [learnScenarioImage.cpp](../prototyping/qlearning/learnScenarioImage.cpp) . It learns the scenario at [ScenarioImage.cpp](../prototyping/qlearning/ScenarioImage.cpp).  This scenario is an empty room, with an apple somewhere, and the agent wins the game by getting the apple.  You can put the apple always in the centre, and place the agent randomly somewhere at the start, or you can put the apple in a random location.

Learning when both the agent and the apple are randomly placed takes a while, a thousand games or so.  When the apple is always in the centre, it's pretty easy, and the agent will learn within a few tens of games.

A representation of the world, and of q, is printed at the end of each game, by the Scenario implementation.

## Python demo

The same world as for the C++ demo is also available in Python, at [python/test_qlearning.py](../python/test_qlearning.py)
* Using some cython-foo, we can also override the Scenario class, just as in C++ :-)
* Since most of the work is being done by the convolutional network, in C++/OpenCL, this runs pretty quickly.  The python Scenario implementation simply has to provide the reward, and the perception, at each move.

