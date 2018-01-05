set -x

echo args: $1

pyenv=$1
echo pyenv: $pyenv

. $HOME/${pyenv}/bin/activate
pip install cython pypandoc pytest numpy || exit 1

pwd
rm -Rf build dist
mkdir build
cd build
cmake .. || exit 1
make -j 4 install || exit 1
cd ..
pwd
# ls
. dist/bin/activate.sh

pwd
cp jenkins/version.txt python
cd python
pwd
rm -Rf dist build DeepCL.egg-info
# ls
pwd
pip uninstall -y DeepCL
python setup.py install || exit 1
if [[ $(uname -p) == x86_64 ]]; then {
    py.test -sv test || exit 1
} fi
python setup.py build_ext -i || exit 1
python setup.py bdist_egg || exit 1

# just ignore the error on next line for now (if already uploaded this version)
python setup.py bdist_egg upload

# just ignore the error on next line for now (if already uploaded this version)
python setup.py sdist register upload

exit 0

