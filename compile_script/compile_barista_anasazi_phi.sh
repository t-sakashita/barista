#!/bin/bash -x

rm -rf build/
mkdir build
cd build

cmake \
-DTRILINOS_PREFIX=${HOME}/lib_try/trilinos-11.0.3-Source/install_build/ \
..

make VERBOSE=1
