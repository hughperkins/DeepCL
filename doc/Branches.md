# Branches

*Main branches:*
* master: what you probably want to fork, if you want to do any changes

*Feature/story branches:*

*Jenkins:*
* jenkins-target: many jenkins build jobs run from this branch (though recently, jenkins builds are triggered by creating a tag, and build the tag, instead)
* jenkins-perf: my Jenkins server runs performance benchmarking tests from this branch
* jenkins-jobs: any commit to the this branch triggers jenkins job builder to automatically update the jenkins jobs from [jenkins/jobs.yaml](../jenkins/jobs.yaml) file

*On-hold feature branches:*
* imagenet-py: partial implementation of python script to read imagenet and train on these
* python_swig: python swig wrappers have been pushed out to a branch, and are no longer part of mainstream
  * note that this is different from the Cython wrappers, which are in the 'python' directory of 'master' branch for example, which are the production Python wrappers for DeepCL
* numpy_i_experiment: a brief experiment on trying to add numpy.i to the python swig wrappers
* try_async_workgroup_copy: they do say that async_workgroup_copy should be the most optimal way of copying between gpu global and local memory.  In my brief experiments with it, in this branch, in fact it was slower.  I just leave it here in case anyone wants to convince themselves

*Probably dead branches:*
* writeweightsmoreoften
* docchanges
* debug-comments

