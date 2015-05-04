# Branches

*Main branches:*
* master: what you probably want to fork, if you want to do any changes
* 3.x.x: previous version.  Just here so the old documentation is available really

*Feature/story branches:*

*github pages:*
* gh-pages: holds the website at http://hughperkins.github.io/DeepCL

*Jenkins:*
* jenkins-target: my Jenkins server builds off this branch, so anything I want jenkins to build, I push to this branch, and then trigger a Jenkins build.  eg, for a release
* benchmarking-jenkins: my Jenkins server runs performance benchmarking tests from this branch

*On-hold feature branches:*
* python_swig: python swig wrappers have been pushed out to a branch, and are no longer part of mainstream
  * note that this is different from the Cython wrappers, which are in the 'python' directory of 'master' branch for example, which are the production Python wrappers for DeepCL
* numpy_i_experiment: a brief experiment on trying to add numpy.i to the python swig wrappers
* try_async_workgroup_copy: they do say that async_workgroup_copy should be the most optimal way of copying between gpu global and local memory.  In my brief experiments with it, in this branch, in fact it was slower.  I just leave it here in case anyone wants to convince themselves

*Probably dead branches:*
* writeweightsmoreoften
* docchanges
* debug-comments

