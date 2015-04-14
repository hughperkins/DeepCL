#!/bin/bash

# bash script to purge everything, so we can rebuild from scratch
# not supported on Windows, clearly :-)


rm -Rf build dist DeepCL.egg-info mysrc *.pyc PyDeepCL.cpp PyDeepCL.pyd *.so

