#!/bin/bash

~/envs/bin/python setup.py bdist_egg upload
~/env-34/bin/python setup.py bdist_egg upload
~/env-34/bin/python setup.py sdist upload 
 
