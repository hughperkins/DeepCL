#!/bin/bash

. $HOME/env-2.7/bin/activate
pip install libyaml
pip install python-jenkins

jenkins-jobs update jenkins/jobs.yaml

