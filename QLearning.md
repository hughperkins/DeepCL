<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Q-learning (draft)](#q-learning-draft)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Q-learning (draft)

* Started to write a q-learning module
* Ideally, this will be callable from Python in the future, via [PyDeepCL](https://pypi.python.org/pypi/PyDeepCL)
* Look at [ScenarioImage.h](prototyping/qlearning/ScenarioImage.h) and [ScenarioImage.cpp](prototyping/qlearning/ScenarioImage.cpp) for an example scenario we can feed to it
  * [learnScenarioImage.cpp](prototyping/qlearning/learnScenarioImage.cpp) is a corresponding example of how we can learn this
* The qlearning module is at [QLearner.h](qlearning/QLearner.h), and the interface for scenarios at [Scenario.h](qlearning/Scenario.h)


