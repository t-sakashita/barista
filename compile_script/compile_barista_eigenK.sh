#!/bin/bash -x

rm -rf build/
mkdir build
cd build
cmake \
-DEIGEN_S_LIB="${HOME}/lib/eigenK/libEigen_s.a;-SCALAPACK;-SSL2BLAMP" \
-D CMAKE_CXX_FLAGS="-Qt" \
-D CMAKE_C_FLAGS="-Qt" \
..

#-DCMAKE_CXX_COMPILER=mpiFCCpx -DCMAKE_CXX_FLAGS="-w -Xg -Kfast,ocl,ilfun -KPIC" \

#-DCMAKE_C_COMPILER=mpifccpx -DCMAKE_C_FLAGS="-w -Xg -Kfast,ocl,ilfunc -KPIC" \

#-DEIGEN_S_LIB='${HOME}/lib/eigenK/libEigen_s.a -SSL2 -SSL2BLAMP'
#-DCMAKE_EXE_LINKER_FLAGS="${HOME}/lib/eigenK/libEigen_s.a -SSL2 -SSL2BLAMP" -DCMAKE_SHARED_LINKER_FLAGS="${HOME}/lib/eigenK/libEigen_s.a -SSL2 -SSL2BLAMP" \

make VERBOSE=1
