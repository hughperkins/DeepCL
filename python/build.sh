#!/bin/bash

# for maintainer's usage only
# no documentation (though raise an issue to ask for some if you want :-) )

rm -Rf build dist *.egg-info PyDeepCL.cxx *.so
(cd ../build; make -j 4 install)
pip uninstall -y DeepCL
python setup.py install || exit 1

