echo args: $1

pyenv=$1
echo pyenv: $pyenv

pwd
cd python
pwd
rm -Rf dist mysrc build PyDeepCL.cpp
ls
. $HOME/env-2.7/bin/activate
pwd
pip install cython pypandoc || exit 1
python setup.py build_ext -i || exit 1
python setup.py bdist_egg || exit 1

# just ignore the error on next line for now (if already uploaded this version)
python setup.py bdist_egg upload

# just ignore the error on next line for now (if already uploaded this version)
python setup.py sdist upload

exit 0

