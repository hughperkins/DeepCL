#!/bin/bash

rm -Rf build DeepCL.egg-info dist
python setup.py install
python test_deepcl.py /norep/data/mnist

