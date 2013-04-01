#!/bin/bash -x

-DEIGEN_S_LIB="/opt/nano/rokko/lib/libEigen_s.a;-lmkl_scalapack_lp64;-lmkl_blacs_intelmpi_lp64;-lmkl_intel_lp64;-lmkl_intel_thread;-lmkl_core;-liomp5;-lpthread" \

rm -rf build/
mkdir build
cd build
cmake \
-DEIGEN_S_INC="/opt/nano/rokko/include" \
-DEIGEN_S_LIB="/opt/nano/rokko/lib/libEigen_s.a -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread" \
-DTRILINOS_PREFIX="${HOME}/lib/trilinos-11.0.3-Source/install_build" \
-D CMAKE_CXX_FLAGS="-Qt" \
-D CMAKE_C_FLAGS="-Qt" \
..

#-DCMAKE_CXX_COMPILER=mpiFCCpx -DCMAKE_CXX_FLAGS="-w -Xg -Kfast,ocl,ilfun -KPIC" \

#-DCMAKE_C_COMPILER=mpifccpx -DCMAKE_C_FLAGS="-w -Xg -Kfast,ocl,ilfunc -KPIC" \

#-DEIGEN_S_LIB='${HOME}/lib/eigenK/libEigen_s.a -SSL2 -SSL2BLAMP'


make VERBOSE=1
