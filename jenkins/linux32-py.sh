echo args: $1

pyenv=$1
echo pyenv: $pyenv

pwd
cd python
pwd
rm -Rf dist mysrc build PyDeepCL.cpp
ls
schroot -c trusty_i386 -- $HOME/${pyenv}/bin/pip install cython pypandoc || exit 1
schroot -c trusty_i386 -- $HOME/${pyenv}/bin/python setup.py build_ext -i || exit 1
schroot -c trusty_i386 -- $HOME/${pyenv}/bin/python setup.py bdist_egg || exit 1

# just ignore the error on next line for now (if already uploaded this version)
schroot -c trusty_i386 -- $HOME/${pyenv}/bin/python setup.py bdist_egg upload
exit 0

