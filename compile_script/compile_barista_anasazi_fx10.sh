#!/bin/bash -x

rm -rf build/
mkdir build
cd build

cmake \
-DTRILINOS_PREFIX=${HOME}/lib/trilinos-11.0.3-Source/install_build/ \
-D CMAKE_CXX_FLAGS="-Qt" \
-D CMAKE_C_FLAGS="-Qt" \
..

make VERBOSE=1
