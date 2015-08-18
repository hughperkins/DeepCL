#!/bin/bash

bin_dir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
dist_dir=$(dirname ${bin_dir})

export PATH=${bin_dir}:$PATH
export LD_LIBRARY_PATH=${dist_dir}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=${dist_dir}/lib:${PYTHONPATH}

